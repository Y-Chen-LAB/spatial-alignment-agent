from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from paper2bench.core.task_card import TaskCard, compile_task_card
from paper2bench.core.validation import validate_task_file


class TaskCardCompilerTests(unittest.TestCase):
    def test_compile_minimal_spatial_alignment_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            card_path = root / "card.json"
            card_path.write_text(
                json.dumps(
                    {
                        "task_id": "demo-task",
                        "domain": "spatial_genomics",
                        "modality": "spatial_transcriptomics",
                        "task_type": "serial_alignment",
                        "input_contract": "h5ad_spatial_with_region_v1",
                        "output_contract": "aligned_coordinate_csv_v1",
                        "metric_profile": "sabench_light_alignment_v1",
                        "instruction": "Align the slices.",
                        "assets": [
                            {"asset_id": "a", "path": str(root / "a.h5ad"), "role": "slice"},
                            {"asset_id": "b", "path": str(root / "b.h5ad"), "role": "slice"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (root / "a.h5ad").write_text("placeholder", encoding="utf-8")
            (root / "b.h5ad").write_text("placeholder", encoding="utf-8")

            compiled = compile_task_card(card_path, root / "demo-task")
            self.assertTrue(compiled.task_toml.exists())
            self.assertTrue(compiled.instruction_md.exists())
            self.assertTrue(compiled.data_manifest.exists())

            task_toml = compiled.task_toml.read_text(encoding="utf-8")
            self.assertIn('schema_version = "paper2bench.task.v0.1"', task_toml)
            self.assertIn('slices = ["a=', task_toml)

            report = validate_task_file(compiled.task_toml, strict_data=True)
            self.assertTrue(report.ok, report.to_json())

    def test_rejects_runner_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            card_path = Path(tmp) / "card.json"
            card_path.write_text(
                json.dumps(
                    {
                        "task_id": "bad-task",
                        "domain": "spatial_genomics",
                        "task_type": "serial_alignment",
                        "model": "gpt-5-mini",
                        "assets": [{"asset_id": "a", "path": "a.h5ad"}],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "model-free"):
                TaskCard.from_path(card_path)


if __name__ == "__main__":
    unittest.main()
