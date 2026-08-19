"""Read-only storage report for PronunciaBench experiment artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "byt5-cmudict-001"
PROJECT_DIR_NAMES = ("experiments", "data", "reports", "notebooks")


def format_bytes(size: int) -> str:
    """Format a byte count with a binary unit."""
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def path_size(path: Path) -> int:
    """Return the size of a file tree without following directory symlinks."""
    if path.is_symlink():
        return path.lstat().st_size
    if path.is_file():
        return path.stat().st_size
    if not path.exists():
        return 0

    total = 0
    pending = [path]
    while pending:
        current = pending.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_symlink():
                            total += entry.stat(follow_symlinks=False).st_size
                        elif entry.is_dir(follow_symlinks=False):
                            pending.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def _read_run_status(run_dir: Path) -> str:
    for name in ("results.json", "provenance.json"):
        path = run_dir / name
        if not path.is_file():
            continue
        try:
            payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        status = payload.get("status") or payload.get("run", {}).get("status")
        if isinstance(status, str):
            return status
    if run_dir.name.lower().startswith("canary_"):
        return "CANARY_RUN"
    return "PARTIAL_RUN"


def collect_report(project_root: Path, experiment_dir: Path) -> dict[str, Any]:
    """Collect project-local storage facts without modifying the filesystem."""
    project_root = project_root.resolve()
    experiment_dir = experiment_dir.resolve()
    volume_probe = experiment_dir
    while not volume_probe.exists():
        if volume_probe.parent == volume_probe:
            raise FileNotFoundError(f"Experiment volume is unavailable: {experiment_dir.anchor}")
        volume_probe = volume_probe.parent
    disk = shutil.disk_usage(volume_probe)

    local_paths = [project_root / name for name in PROJECT_DIR_NAMES]
    local_paths.extend(sorted(project_root.glob(".venv*")))
    project_sizes = {
        str(path.relative_to(project_root)): path_size(path)
        for path in local_paths
        if path.exists()
    }

    checkpoints = []
    if experiment_dir.exists():
        for checkpoint in sorted(experiment_dir.rglob("checkpoint-*")):
            if checkpoint.is_dir():
                checkpoints.append({"path": str(checkpoint), "bytes": path_size(checkpoint)})

    runs = []
    reclaimable_bytes = 0
    if experiment_dir.exists():
        for run_dir in sorted(path for path in experiment_dir.iterdir() if path.is_dir()):
            status = _read_run_status(run_dir)
            run_checkpoints = [item for item in checkpoints if Path(item["path"]).parent == run_dir]
            checkpoint_bytes = sum(item["bytes"] for item in run_checkpoints)
            potentially_removable = status in {"CANARY_RUN", "PARTIAL_RUN"}
            if potentially_removable:
                reclaimable_bytes += checkpoint_bytes
            runs.append(
                {
                    "path": str(run_dir),
                    "status": status,
                    "bytes": path_size(run_dir),
                    "checkpoint_count": len(run_checkpoints),
                    "checkpoint_bytes": checkpoint_bytes,
                    "potentially_removable_checkpoint_bytes": (
                        checkpoint_bytes if potentially_removable else 0
                    ),
                }
            )

    return {
        "project_root": str(project_root),
        "experiment_dir": str(experiment_dir),
        "volume": experiment_dir.anchor or project_root.anchor,
        "free_bytes": disk.free,
        "project_sizes": project_sizes,
        "checkpoints": checkpoints,
        "runs": runs,
        "reclaimable_bytes": reclaimable_bytes,
    }


def print_report(report: dict[str, Any]) -> None:
    """Print a human-readable storage report."""
    print("PronunciaBench Storage Report")
    print()
    print(f"Project: {report['project_root']}")
    print(f"Current experiment output directory: {report['experiment_dir']}")
    print(f"Volume: {report['volume']}")
    print(f"Free: {format_bytes(report['free_bytes'])}")
    print()
    print("Project-local directories:")
    if report["project_sizes"]:
        for name, size in report["project_sizes"].items():
            print(f"  {name}/: {format_bytes(size)}")
    else:
        print("  (none found)")

    print()
    print(f"Checkpoints: {len(report['checkpoints'])}")
    for checkpoint in report["checkpoints"]:
        print(f"  {checkpoint['path']}: {format_bytes(checkpoint['bytes'])}")

    candidates = [
        run for run in report["runs"] if run["potentially_removable_checkpoint_bytes"] > 0
    ]
    print()
    print(f"Canary/partial runs: {sum(run['status'] != 'COMPLETED' for run in report['runs'])}")
    for run in report["runs"]:
        print(
            f"  {run['path']} [{run['status']}]: {format_bytes(run['bytes'])}, "
            f"{run['checkpoint_count']} checkpoint(s)"
        )
    print()
    print("Potentially removable canary/partial checkpoints:")
    if candidates:
        for run in candidates:
            print(
                f"  {run['path']}: "
                f"{format_bytes(run['potentially_removable_checkpoint_bytes'])}"
            )
    else:
        print("  (none)")
    print(
        "Total reclaimable project-local bytes: "
        f"{format_bytes(report['reclaimable_bytes'])}"
    )
    print("Read-only report: no files were deleted or changed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="PronunciaBench repository root",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help="Experiment root to inspect (default: experiments/byt5-cmudict-001)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    experiment_dir = args.experiment_dir or Path("experiments") / EXPERIMENT_ID
    if not experiment_dir.is_absolute():
        experiment_dir = project_root / experiment_dir
    report = collect_report(project_root, experiment_dir)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
