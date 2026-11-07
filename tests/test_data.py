from __future__ import annotations

import unittest

import pandas as pd

from gdt.data import PairDataset


class PairDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = PairDataset(
            pd.DataFrame(
                [
                    ["Alice Smith", "Smith, Alice"],
                    ["Bob Jones", "Jones, Bob"],
                    ["Carol White", "White, Carol"],
                ]
            )
        )

    def test_id_mapping_is_explicit_and_reversible(self) -> None:
        self.assertEqual(self.dataset.source_id(1), 1)
        self.assertEqual(self.dataset.target_id(1), 4)
        self.assertEqual(self.dataset.row_id_from_target(4), 1)
        self.assertEqual(self.dataset.paired_id(1), 4)
        self.assertEqual(self.dataset.paired_id(4), 1)
        self.assertEqual(self.dataset.text_for_id(4), "Jones, Bob")

    def test_clustering_values_include_only_example_targets(self) -> None:
        self.assertEqual(
            self.dataset.clustering_values(2),
            ["Alice Smith", "Bob Jones", "Smith, Alice", "Jones, Bob"],
        )

    def test_short_dataset_has_no_test_rows(self) -> None:
        self.assertEqual(self.dataset.test_sources(1), {})
        self.assertEqual(self.dataset.ground_truth(1), {})

    def test_all_counts_reserve_five_pairs_and_preserve_original_ids(self) -> None:
        dataset = PairDataset(pd.DataFrame([(f"input {i}", f"output {i}") for i in range(8)]))
        for count in range(6):
            with self.subTest(count=count):
                self.assertEqual(list(dataset.test_sources(count)), [5, 6, 7])
                self.assertEqual(dataset.ground_truth(count),
                                 {i: f"output {i}" for i in [5, 6, 7]})
                self.assertEqual(list(dataset.clustering_mapping(count)),
                                 list(range(count)) + [5, 6, 7] + list(range(8, 8 + count)))
        with self.assertRaisesRegex(ValueError, "exceed 5"):
            dataset.validate_example_count(6)

    def test_invalid_schema_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly two"):
            PairDataset(pd.DataFrame([["one column"]]))
        with self.assertRaisesRegex(ValueError, "missing"):
            PairDataset(pd.DataFrame([["source", None]]))


if __name__ == "__main__":
    unittest.main()
