from __future__ import annotations

import unittest

import pandas as pd

from genomics_agent_bench.ranking import rank_sabench_style_cohort


class SABenchStyleRankTests(unittest.TestCase):
    def test_ranks_with_metric_direction_and_replicates(self) -> None:
        scores = pd.DataFrame(
            [
                {"method": "A", "task": "t1", "gene_pcc": 0.9, "mae": 3.0},
                {"method": "A", "task": "t1", "gene_pcc": 0.7, "mae": 1.0},
                {"method": "B", "task": "t1", "gene_pcc": 0.6, "mae": 3.0},
                {"method": "A", "task": "t2", "gene_pcc": 0.8, "mae": 2.0},
                {"method": "B", "task": "t2", "gene_pcc": 0.9, "mae": 1.0},
            ]
        )
        result = rank_sabench_style_cohort(scores, metrics=["gene_pcc", "mae"])

        ranks = result.task_metric_ranks
        a_t1_pcc = ranks.loc[(ranks.method == "A") & (ranks.task == "t1") & (ranks.metric == "gene_pcc"), "rank"].item()
        a_t1_mae = ranks.loc[(ranks.method == "A") & (ranks.task == "t1") & (ranks.metric == "mae"), "rank"].item()
        self.assertEqual(a_t1_pcc, 1.0)  # mean replicate PCC = 0.8 > B's 0.6
        self.assertEqual(a_t1_mae, 1.0)  # mean replicate MAE = 2.0 < B's 3.0
        self.assertEqual(result.overall_summary.iloc[0]["method"], "A")
        self.assertEqual(result.overall_summary.iloc[0]["sabench_style_overall_rank"], 1)

    def test_rejects_local_composite(self) -> None:
        scores = pd.DataFrame([{"method": "A", "task": "t1", "overall_mean": 0.9}])
        with self.assertRaisesRegex(ValueError, "overall_mean"):
            rank_sabench_style_cohort(scores, metrics=["overall_mean"])

    def test_duplicate_rows_require_explicit_aggregation_when_disabled(self) -> None:
        scores = pd.DataFrame(
            [
                {"method": "A", "task": "t1", "gene_pcc": 0.9},
                {"method": "A", "task": "t1", "gene_pcc": 0.8},
            ]
        )
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            rank_sabench_style_cohort(scores, metrics=["gene_pcc"], aggregate_replicates="none")


if __name__ == "__main__":
    unittest.main()
