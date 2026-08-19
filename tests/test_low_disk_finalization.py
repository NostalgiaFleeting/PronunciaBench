"""Tests for the low-disk training and frozen-safe finalization gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
import byt5_experiment as experiment  # noqa: E402
import storage_report  # noqa: E402


def parse_args(*arguments: str) -> Any:
    return experiment.build_parser().parse_args(list(arguments))


def write_split(directory: Path, name: str, size: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{name}.jsonl").open("w", encoding="utf-8") as file_handle:
        for index in range(size):
            row = {"word": f"word{index}", "pronunciation": "AH0", "label": f"WORD{index}"}
            file_handle.write(json.dumps(row) + "\n")


def test_low_disk_full_run_uses_epoch_checkpoints(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    args = parse_args("--low-disk", "--amp-backend", "fp32")
    parameters = set(experiment.inspect.signature(experiment.TrainingArguments.__init__).parameters)
    parameters.add("save_safetensors")
    monkeypatch.setattr(experiment, "_training_argument_parameters", lambda: parameters)

    kwargs = experiment.build_training_argument_kwargs(args, tmp_path, False, False)

    eval_key = "eval_strategy" if "eval_strategy" in parameters else "evaluation_strategy"
    assert kwargs[eval_key] == "epoch"
    assert kwargs["save_strategy"] == "epoch"
    assert kwargs["save_total_limit"] == 2
    assert kwargs["save_only_model"] is False
    assert kwargs["save_safetensors"] is True
    assert kwargs["load_best_model_at_end"] is True
    assert "save_steps" not in kwargs


def test_canary_saves_its_final_step_and_is_not_best_model_tracking(tmp_path: Path):
    args = parse_args("--max-steps", "2", "--low-disk")
    kwargs = experiment.build_training_argument_kwargs(args, tmp_path, False, False)

    assert experiment.is_canary_run(args) is True
    assert kwargs["save_strategy"] == "steps"
    assert kwargs["save_steps"] == 2
    assert kwargs["save_total_limit"] == 1
    assert kwargs["load_best_model_at_end"] is False
    assert kwargs.get("eval_strategy", kwargs.get("evaluation_strategy")) == "no"


def test_oov_metrics_preserve_legacy_result_contract():
    predictions = [
        {
            "word": "unseen",
            "per_with_stress": 0.123456,
            "per_without_stress": 0.234567,
            "exact_match": False,
        }
    ]

    metrics = experiment.compute_oov_metrics(predictions, [{"word": "seen"}])

    assert metrics == {
        "oov_words": 1,
        "oov_per_with_stress": 0.1235,
        "oov_per_without_stress": 0.2346,
        "oov_exact_match": 0.0,
        "note": "1 OOV words out of 1 test words",
    }


def test_reliability_metrics_preserve_legacy_rounding_and_calibration_interface():
    predictions = [
        {
            "word": "exact-length",
            "reference": "AH0 B",
            "prediction": "AH0 D",
            "per_with_stress": 0.123456,
            "exact_match": False,
        },
        {
            "word": "short",
            "reference": "AH0 B",
            "prediction": "AH0",
            "per_with_stress": 0.5,
            "exact_match": False,
        },
    ]

    metrics = experiment.evaluate_reliability_layer(
        predictions,
        [{"word": "calibration", "pronunciation": "K AE1 L"}],
    )

    assert metrics["bucket_analysis"] == [
        {
            "bucket": "confidence_2_3",
            "sample_count": 1,
            "mean_per": 0.5,
            "exact_match_rate": 0.0,
        },
        {
            "bucket": "confidence_4_5",
            "sample_count": 1,
            "mean_per": 0.1235,
            "exact_match_rate": 0.0,
        },
    ]
    assert metrics["monotonicity_preserved"] is True
    assert metrics["abstention_results"][0] == {
        "threshold": 0.3,
        "coverage": 1.0,
        "selective_per": 0.3117,
        "abstention_rate": 0.0,
    }


@pytest.mark.parametrize("with_finalization_smoke", [False, True])
def test_canary_pretraining_never_loads_frozen_test(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    with_finalization_smoke: bool,
):
    data_dir = tmp_path / "data"
    write_split(data_dir, "train", 3)
    write_split(data_dir, "validation", 3)
    (data_dir / "test.jsonl").write_text("this must never be parsed", encoding="utf-8")
    arguments = ["--data-dir", str(data_dir), "--max-steps", "2"]
    if with_finalization_smoke:
        arguments.extend(("--finalization-smoke", "2"))
    args = parse_args(*arguments)
    original_load_split = experiment.load_split

    def frozen_test_trap(path: Path):
        if path.name == "test.jsonl":
            raise AssertionError("FrozenTestAccessError")
        return original_load_split(path)

    monkeypatch.setattr(experiment, "load_split", frozen_test_trap)
    splits = experiment.load_training_splits(args)
    validation = splits.get("validation", [])
    split_name, rows, frozen_test_opened = experiment.select_finalization_data(
        args,
        validation,
    )

    assert frozen_test_opened is False
    if with_finalization_smoke:
        assert split_name == "validation"
        assert len(rows) == 2
    else:
        assert split_name is None
        assert rows == []
        assert set(splits) == {"train"}


def test_atomic_json_failure_preserves_previous_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    artifact = tmp_path / "results.json"
    artifact.write_text('{"status":"old"}\n', encoding="utf-8")

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(experiment.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        experiment.atomic_write_json(artifact, {"status": "new"})

    assert json.loads(artifact.read_text(encoding="utf-8")) == {"status": "old"}
    assert list(tmp_path.glob("*.tmp")) == []
    assert [path for path in tmp_path.iterdir() if path.name.startswith(".results.json.")] == []


def test_cleanup_rejects_paths_outside_active_run(tmp_path: Path):
    run_dir = tmp_path / "experiment" / "canary_1"
    inside = run_dir / "checkpoint-2"
    outside = tmp_path / "checkpoint-99"
    inside.mkdir(parents=True)
    outside.mkdir()

    with pytest.raises(ValueError, match="Unsafe cleanup target"):
        experiment.validate_cleanup_targets([inside, outside], run_dir)

    assert inside.exists()
    assert outside.exists()


def test_no_checkpoint_cleanup_after_failure(tmp_path: Path):
    run_dir = tmp_path / "canary_1"
    checkpoint = run_dir / "checkpoint-2"
    checkpoint.mkdir(parents=True)
    args = parse_args("--max-steps", "2", "--cleanup-canary")

    reclaimed = experiment.run_post_success_cleanup(False, args, run_dir, None)

    assert reclaimed == 0
    assert checkpoint.is_dir()


def test_successful_compaction_preserves_final_artifacts(tmp_path: Path):
    run_dir = tmp_path / "run_1"
    checkpoint = run_dir / "checkpoint-10"
    best_model = run_dir / "best_model"
    checkpoint.mkdir(parents=True)
    best_model.mkdir()
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer-state")
    (best_model / "config.json").write_text("{}", encoding="utf-8")
    (best_model / "model.safetensors").write_bytes(b"final-weights")
    for artifact in (
        "results.json",
        "provenance.json",
        "error_analysis.csv",
        "training_log.json",
        "protocol_references.json",
    ):
        (run_dir / artifact).write_text("preserve", encoding="utf-8")
    args = parse_args("--compact-after-success")

    reclaimed = experiment.run_post_success_cleanup(True, args, run_dir, best_model)

    assert reclaimed == len(b"optimizer-state")
    assert not checkpoint.exists()
    assert (best_model / "model.safetensors").read_bytes() == b"final-weights"
    assert (run_dir / "results.json").read_text(encoding="utf-8") == "preserve"


def test_cleanup_provenance_is_written_atomically(tmp_path: Path):
    run_dir = tmp_path / "canary_1"
    run_dir.mkdir()
    (run_dir / "provenance.json").write_text(
        json.dumps({"storage": {}, "finalization": {"complete": True}}),
        encoding="utf-8",
    )

    experiment.record_cleanup_provenance(run_dir, "cleanup_canary", 1234)

    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    cleanup = provenance["storage"]["post_success_cleanup"]
    assert cleanup["mode"] == "cleanup_canary"
    assert cleanup["reclaimed_bytes"] == 1234
    assert cleanup["checkpoint_count_after"] == 0


def test_cleanup_canary_never_accepts_a_full_run(tmp_path: Path):
    run_dir = tmp_path / "run_1"
    (run_dir / "checkpoint-2").mkdir(parents=True)
    args = parse_args()
    args.cleanup_canary = True

    with pytest.raises(ValueError, match="can only remove checkpoints from a canary"):
        experiment.run_post_success_cleanup(True, args, run_dir, None)


class FakeExportModel:
    def __init__(self) -> None:
        self.calls = 0

    def save_pretrained(self, path: str, safe_serialization: bool = False) -> None:
        assert safe_serialization is True
        self.calls += 1
        destination = Path(path)
        (destination / "config.json").write_text("{}", encoding="utf-8")
        (destination / "model.safetensors").write_bytes(b"weights")


class FakeExportTokenizer:
    def save_pretrained(self, path: str) -> None:
        (Path(path) / "tokenizer_config.json").write_text("{}", encoding="utf-8")


def test_best_model_is_exported_once(tmp_path: Path):
    model = FakeExportModel()
    tokenizer = FakeExportTokenizer()

    exported = experiment.export_best_model_once(model, tokenizer, tmp_path)

    assert exported == tmp_path / "best_model"
    assert (exported / "model.safetensors").is_file()
    assert model.calls == 1
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        experiment.export_best_model_once(model, tokenizer, tmp_path)
    assert model.calls == 1


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text: str, **_kwargs: Any) -> dict[str, torch.Tensor]:
        token = 2 if text else 0
        return {
            "input_ids": torch.tensor([[token, 1]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        }

    def batch_decode(self, values: Any, **_kwargs: Any) -> list[str]:
        return ["AH0" for _ in range(len(values))]


class FakeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(use_cache=False)
        self.checkpointing_disabled = False

    def to(self, _device: torch.device) -> FakeModel:
        return self

    def eval(self) -> FakeModel:
        return self

    def gradient_checkpointing_disable(self) -> None:
        self.checkpointing_disabled = True

    def generate(self, input_ids: torch.Tensor, **_kwargs: Any) -> torch.Tensor:
        return torch.ones((input_ids.shape[0], 2), dtype=torch.long)


class FakeTrainer:
    def __init__(self, run_dir: Path) -> None:
        checkpoint = run_dir / "checkpoint-2"
        self.run_dir = run_dir
        self.checkpoint = checkpoint
        self.state = SimpleNamespace(
            best_model_checkpoint=str(checkpoint),
            log_history=[{"step": 2, "loss": 1.0}],
        )
        self.optimizer = object()
        self.lr_scheduler = object()
        self.model = object()

    def train(self, **_kwargs: Any) -> Any:
        self.checkpoint.mkdir()
        (self.checkpoint / "model.safetensors").write_bytes(b"checkpoint")
        return SimpleNamespace(training_loss=1.0)


def test_fake_finalization_smoke_runs_complete_pipeline_without_test_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    data_dir = tmp_path / "data"
    write_split(data_dir, "train", 4)
    write_split(data_dir, "validation", 3)
    (data_dir / "test.jsonl").write_text("frozen-test-trap", encoding="utf-8")
    experiment_root = tmp_path / "experiments"
    arguments = (
        "--data-dir",
        str(data_dir),
        "--experiment-dir",
        str(experiment_root),
        "--max-steps",
        "2",
        "--finalization-smoke",
        "2",
        "--low-disk",
        "--amp-backend",
        "fp32",
    )
    args = parse_args(*arguments)
    original_load_split = experiment.load_split

    def frozen_test_trap(path: Path):
        if path.name == "test.jsonl":
            raise AssertionError("FrozenTestAccessError")
        return original_load_split(path)

    monkeypatch.setattr(experiment, "load_split", frozen_test_trap)
    monkeypatch.setattr(
        experiment,
        "build_protocol_references",
        lambda strict: {"strict": strict},
    )
    monkeypatch.setattr(
        experiment,
        "print_gpu_diagnostics",
        lambda: {
            "gpu_available": False,
            "pytorch_version": torch.__version__,
            "cuda_version": "N/A",
            "gpu_name": "CPU",
            "vram_gb": 0.0,
            "compute_capability": None,
            "bf16_supported": False,
            "fp16_supported": False,
        },
    )
    tokenizer = FakeTokenizer()
    monkeypatch.setattr(
        experiment.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: tokenizer,
    )
    monkeypatch.setattr(
        experiment.AutoModelForSeq2SeqLM,
        "from_pretrained",
        lambda *_args, **_kwargs: FakeModel(),
    )

    def fake_create_trainer(
        _args: Any,
        _model: Any,
        _tokenizer: Any,
        _train_dataset: Any,
        _validation_dataset: Any,
        run_dir: Path,
        _use_bf16: bool,
        _use_fp16: bool,
    ) -> tuple[FakeTrainer, dict[str, Any]]:
        return FakeTrainer(run_dir), {
            "save_strategy": "steps",
            "save_total_limit": 1,
            "save_only_model": False,
        }

    monkeypatch.setattr(experiment, "create_trainer", fake_create_trainer)
    monkeypatch.setattr(
        experiment,
        "load_checkpoint_model",
        lambda _checkpoint, _device: FakeModel(),
    )
    monkeypatch.setattr(experiment, "get_git_sha", lambda: "deadbeef")

    outcome = experiment.run_experiment(args)

    run_dir = outcome["run_dir"]
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    output = capsys.readouterr().out
    assert outcome["frozen_test_opened"] is False
    assert results["status"] == "CANARY_RUN"
    assert results["evaluation_results"]["n_samples"] == 2
    assert results["generation_throughput"]["estimated_frozen_test_samples"] == 6838
    assert results["generation_throughput"]["estimated_frozen_test_seconds"] > 0
    assert "actual frozen-test duration is not guaranteed" in (
        results["generation_throughput"]["estimate_note"]
    )
    assert provenance["finalization"]["complete"] is True
    assert (run_dir / "error_analysis.csv").is_file()
    assert (run_dir / "checkpoint-2").is_dir()
    assert not (run_dir / "best_model").exists()
    assert "Frozen test opened: NO" in output
    assert "[FINALIZE 8/8] Finalization complete" in output


def test_storage_preflight_blocks_before_training(tmp_path: Path):
    with pytest.raises(RuntimeError, match="Training has NOT started"):
        experiment.inspect_output_disk(tmp_path / "new-output", min_free_gb=10**9)


def test_canonical_split_hash_matches_manifest_representation(tmp_path: Path):
    split_path = tmp_path / "test.jsonl"
    rows = [
        {"word": "BETA", "pronunciation": "B EY1 T AH0", "label": "BETA"},
        {"word": "ALPHA", "pronunciation": "AE1 L F AH0", "label": "ALPHA"},
    ]
    with split_path.open("w", encoding="utf-8") as file_handle:
        for row in rows:
            file_handle.write(json.dumps(row) + "\n")
    expected = "ALPHA\tAE1 L F AH0\tALPHA\nBETA\tB EY1 T AH0\tBETA"
    expected_hash = experiment.hashlib.sha256(expected.encode()).hexdigest()

    actual_hash, words, variants = experiment.compute_canonical_split_sha256(
        experiment.load_split(split_path)
    )

    assert actual_hash == expected_hash
    assert words == 2
    assert variants == 2


def canonical_lf_payload(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def test_canonical_text_hash_normalizes_lf_crlf_and_lone_cr(tmp_path: Path):
    manifest_lf = canonical_lf_payload(experiment.TEST_MANIFEST_PATH)
    assert experiment.hashlib.sha256(manifest_lf).hexdigest() == (
        experiment.EXPECTED_TEST_MANIFEST_SHA256
    )

    representations = {
        "lf.json": manifest_lf,
        "crlf.json": manifest_lf.replace(b"\n", b"\r\n"),
        "cr.json": manifest_lf.replace(b"\n", b"\r"),
    }
    for name, payload in representations.items():
        path = tmp_path / name
        path.write_bytes(payload)
        assert experiment.sha256_canonical_text(path) == (
            experiment.EXPECTED_TEST_MANIFEST_SHA256
        )


def test_canonical_text_hash_detects_substantive_change(tmp_path: Path):
    manifest_lf = canonical_lf_payload(experiment.TEST_MANIFEST_PATH)
    modified = manifest_lf.replace(b'"n_lexical_items": 5881', b'"n_lexical_items": 5882')
    assert modified != manifest_lf
    path = tmp_path / "modified-test-manifest.json"
    path.write_bytes(modified)

    assert experiment.sha256_canonical_text(path) != experiment.EXPECTED_TEST_MANIFEST_SHA256


@pytest.mark.parametrize("line_ending", [b"\n", b"\r\n"], ids=["lf", "crlf"])
def test_strict_protocol_validation_accepts_lf_and_crlf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    line_ending: bytes,
):
    protocol_path = tmp_path / "experiment_protocol.json"
    manifest_path = tmp_path / "test_manifest.json"
    protocol_path.write_bytes(
        canonical_lf_payload(experiment.EXPERIMENT_PROTOCOL_PATH).replace(b"\n", line_ending)
    )
    manifest_path.write_bytes(
        canonical_lf_payload(experiment.TEST_MANIFEST_PATH).replace(b"\n", line_ending)
    )
    monkeypatch.setattr(experiment, "EXPERIMENT_PROTOCOL_PATH", protocol_path)
    monkeypatch.setattr(experiment, "TEST_MANIFEST_PATH", manifest_path)

    references = experiment.build_protocol_references(strict=True)

    assert references["experiment_protocol"]["matches_frozen_reference"] is True
    assert references["test_manifest"]["matches_frozen_reference"] is True
    for reference in references.values():
        assert reference["hash_algorithm"] == "sha256"
        assert reference["hash_semantics"] == "canonical-text-lf-v1"
        assert reference["canonical_sha256"] == reference["expected_canonical_sha256"]


def test_strict_protocol_validation_rejects_content_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    protocol_path = tmp_path / "experiment_protocol.json"
    manifest_path = tmp_path / "test_manifest.json"
    protocol_path.write_bytes(canonical_lf_payload(experiment.EXPERIMENT_PROTOCOL_PATH))
    manifest_lf = canonical_lf_payload(experiment.TEST_MANIFEST_PATH)
    manifest_path.write_bytes(
        manifest_lf.replace(b'"n_lexical_items": 5881', b'"n_lexical_items": 5882')
    )
    monkeypatch.setattr(experiment, "EXPERIMENT_PROTOCOL_PATH", protocol_path)
    monkeypatch.setattr(experiment, "TEST_MANIFEST_PATH", manifest_path)

    with pytest.raises(RuntimeError, match="canonical-text-lf-v1"):
        experiment.build_protocol_references(strict=True)


def test_strict_protocol_references_match_repository_files():
    references = experiment.build_protocol_references(strict=True)

    assert references["experiment_protocol"]["matches_frozen_reference"] is True
    assert references["test_manifest"]["matches_frozen_reference"] is True
    assert references["experiment_protocol"]["canonical_sha256"] == (
        experiment.EXPECTED_PROTOCOL_SHA256
    )
    assert references["test_manifest"]["canonical_sha256"] == (
        experiment.EXPECTED_TEST_MANIFEST_SHA256
    )
    assert references["test_manifest"]["legacy_windows_working_tree_sha256"] == (
        experiment.LEGACY_WINDOWS_TEST_MANIFEST_RAW_SHA256
    )


def test_storage_report_is_read_only_and_counts_canary_checkpoints(tmp_path: Path):
    project_root = tmp_path / "project"
    experiment_root = project_root / "experiments" / experiment.EXPERIMENT_ID
    run_dir = experiment_root / "canary_1"
    checkpoint = run_dir / "checkpoint-2"
    checkpoint.mkdir(parents=True)
    payload = b"checkpoint-bytes"
    (checkpoint / "model.safetensors").write_bytes(payload)
    (run_dir / "results.json").write_text(
        json.dumps({"status": "CANARY_RUN"}),
        encoding="utf-8",
    )

    report = storage_report.collect_report(project_root, experiment_root)

    assert report["reclaimable_bytes"] == len(payload)
    assert len(report["checkpoints"]) == 1
    assert report["runs"][0]["status"] == "CANARY_RUN"
    assert (checkpoint / "model.safetensors").read_bytes() == payload
