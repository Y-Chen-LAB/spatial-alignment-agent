from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from genomics_agent_bench.metrics import (
    ALBIS_MNN_METRICS, PAPER_GENE_METRICS, _gene_grid_metrics,
    _spatial_mnn_metrics, score_alignment_task,
)


class FakeSlice:
    def __init__(self, sid, coords, labels, expression):
        self.id = sid
        self.path = Path(sid)
        self.spatial = np.asarray(coords, dtype=float)
        self.obs_names = np.array([f"obs{i}" for i in range(len(coords))])
        self.labels = np.array(labels)
        self.expression = np.asarray(expression, dtype=float)

    def has_obs_column(self, key):
        return key in {"Region", "x_GT", "y_GT"}

    def obs_column(self, key):
        return self.labels if key == "Region" else self.spatial[:, int(key == "y_GT")]

    def expression_for_genes(self, genes):
        return list(genes), self.expression


class AlbisScoringTests(unittest.TestCase):
    def setUp(self):
        self.left = FakeSlice("a", [[0, 0], [10, 0], [0, 10]], ["A", "B", "C"], [[1, 0], [0, 1], [1, 1]])
        self.right = FakeSlice("b", [[0, 0], [10, 0], [0, 10]], ["A", "B", "C"], [[1, 0], [1, 0], [0, 0]])

    def test_mutual_matches_labels_and_coverage_unequal_sizes(self):
        right = FakeSlice("b", [[1, 0], [9, 0]], ["A", "wrong"], [[1, 0], [0, 1]])
        scores, (li, ri) = _spatial_mnn_metrics(self.left, right, self.left.spatial, right.spatial, "Region")
        np.testing.assert_array_equal(li, [0, 1])
        np.testing.assert_array_equal(ri, [0, 1])
        self.assertEqual(scores["mnn_coverage"], 0.8)
        self.assertEqual(scores["mnn_label_accuracy"], 0.5)
        self.assertEqual(scores["mnn_distance_um"], 1.0)

    def test_expression_cosine_and_zero_vector_without_grid_overlap(self):
        # Diagnostic must survive the gene-grid early return for non-overlap.
        scores = _gene_grid_metrics(self.left, self.right, self.left.spatial,
                                    self.right.spatial + 100, ["g1", "g2"], 10,
                                    mnn_indices=(np.arange(3), np.arange(3)))
        self.assertAlmostEqual(scores["mnn_expression_cosine"], 1 / 3)
        self.assertEqual(scores["gene_cosine"], 0)

    def test_profile_scores_split_and_excludes_diagnostics_from_composite(self):
        coords = {s.id: s.spatial.copy() for s in [self.left, self.right]}
        coords["b"][1:] += 100  # errors outside the scored split
        with patch("genomics_agent_bench.metrics.read_coordinate_outputs", return_value=coords):
            scores = score_alignment_task(
                [self.left, self.right], Path("unused"), marker_genes=["g1", "g2"],
                metric_profile="albis_alignment_v1",
                split_masks={"a": np.array([True, False, False]), "b": np.array([True, False, False])},
                score_split="eval",
            )
        aggregate = scores["aggregate"]
        self.assertEqual(aggregate["mae"], 0)
        self.assertEqual(aggregate["mnn_coverage"], 1)
        self.assertEqual(aggregate["mnn_label_accuracy"], 1)
        self.assertEqual(aggregate["mnn_expression_cosine"], 1)
        self.assertTrue(np.isnan(aggregate["mae_reduction"]))  # identity already exact
        self.assertEqual(aggregate["overall_mean"], 0)
        self.assertTrue(set(ALBIS_MNN_METRICS + PAPER_GENE_METRICS).issubset(aggregate))
        self.assertEqual(scores["scoring_rules"]["primary_metric"], "mae")
        self.assertFalse(scores["scoring_rules"]["metrics"]["mnn_coverage"]["included_in_overall_mean"])

    def test_all_materialized_albis_tasks_use_profile(self):
        import tomllib
        root = Path(__file__).resolve().parents[1] / "benchmarks/spatial_alignment/tasks"
        tasks = list(root.glob("sa-3*-albis-*/task.toml"))
        self.assertEqual(len(tasks), 12)
        self.assertFalse(any(path.parent.name.endswith("-smoke") for path in tasks))
        for path in tasks:
            self.assertEqual(tomllib.loads(path.read_text())["metric_profile"], "albis_alignment_v1", str(path))


if __name__ == "__main__":
    unittest.main()
