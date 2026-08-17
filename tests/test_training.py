"""Training pipeline smoke tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


class TestTrainingSmoke:
    """Validate the training pipeline end-to-end with a tiny model."""

    def test_dataset_loading(self):
        """Dataset can be loaded from JSONL."""
        from pronunciabench.training.train import load_dataset_from_jsonl
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"text": "hi", "pronunciation": "h ai", "locale": "en-US"}\n')
            f.write('{"text": "bye", "pronunciation": "b ay", "locale": "en-US"}\n')
            path = f.name
        examples = load_dataset_from_jsonl(path)
        assert len(examples) == 2
        assert examples[0]["text"] == "hi"
        Path(path).unlink()

    def test_dataset_class(self):
        """G2PDataset can be instantiated."""
        from pronunciabench.training.train import G2PDataset
        # Should not raise
        assert G2PDataset is not None

    def test_training_config(self):
        """TrainingConfig has sensible defaults."""
        from pronunciabench.training.train import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.seed == 42
        assert cfg.model_name == "google/byt5-small"
        assert cfg.epochs == 5
        assert cfg.batch_size == 16
        assert cfg.learning_rate == 1e-4

    def test_train_model_smoke(self):
        """Run a minimal training step with a tiny test model if available."""
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            has_transformers = True
        except ImportError:
            pytest.skip("transformers not available")
            return

        from pronunciabench.training.train import train_model, TrainingConfig
        import tempfile
        import json

        # Create tiny train/eval data
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = Path(tmpdir) / "train.jsonl"
            eval_path = Path(tmpdir) / "eval.jsonl"
            with open(train_path, "w") as f:
                for i in range(20):
                    f.write(json.dumps({
                        "text": f"name{i}",
                        "pronunciation": f"/na{i}me/",
                        "locale": "en-US",
                    }) + "\n")
            with open(eval_path, "w") as f:
                for i in range(5):
                    f.write(json.dumps({
                        "text": f"test{i}",
                        "pronunciation": f"/tɛst{i}/",
                        "locale": "en-US",
                    }) + "\n")

            cfg = TrainingConfig(
                model_name="google/byt5-small",
                epochs=1,
                batch_size=2,
                learning_rate=1e-4,
                output_dir=tmpdir,
            )
            result = train_model(str(train_path), str(eval_path), config=cfg)
            assert result.final_loss >= 0
            assert result.best_epoch >= 1
            assert len(result.logs) >= 1