from __future__ import annotations

import unittest

import numpy as np

from genomics_agent_bench.splits import default_split_manifest, split_masks_for_slices


class FakeSlice:
    def __init__(self, slice_id: str, n_obs: int):
        self.id = slice_id
        self.obs_names = np.array([f"obs_{idx}" for idx in range(n_obs)])


class SplitTests(unittest.TestCase):
    def test_sklearn_split_is_deterministic_and_balanced(self) -> None:
        manifest = default_split_manifest("demo", train_fraction=0.5, seed=7)
        slices = [FakeSlice("a", 10), FakeSlice("b", 11)]

        train_1 = split_masks_for_slices(slices, manifest, split_name="train")
        train_2 = split_masks_for_slices(slices, manifest, split_name="train")
        eval_masks = split_masks_for_slices(slices, manifest, split_name="eval")

        np.testing.assert_array_equal(train_1["a"], train_2["a"])
        self.assertEqual(int(train_1["a"].sum()), 5)
        self.assertEqual(int(eval_masks["a"].sum()), 5)
        self.assertEqual(int(train_1["b"].sum()) + int(eval_masks["b"].sum()), 11)
        self.assertFalse(np.any(train_1["b"] & eval_masks["b"]))


if __name__ == "__main__":
    unittest.main()
