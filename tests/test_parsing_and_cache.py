from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from LLMUtil.ResponseParser import RParser, RParser_Prompt
from Utils.CacheUtil import (
    cache_result,
    perfect_cluster_override,
    perfect_split_override,
    perfect_split_override_prompt,
)
from Utils.config import Config


class ParserTests(unittest.TestCase):
    def test_value_response_parsing(self) -> None:
        response = "first name: Alice\n[ITEM]\nlast name: Jones"
        self.assertEqual(RParser.split_items(response), ["first name: Alice", "last name: Jones"])
        self.assertEqual(RParser.get_types_from_split(response), ["first name", "last name"])
        self.assertEqual(
            RParser.parse_component_str(RParser.split_items(response))[0],
            {"first name": "Alice"},
        )

    def test_prompt_response_parsing(self) -> None:
        parsed = RParser_Prompt.parse_component_str(["[first_name]\n[last_name]"])
        self.assertEqual(parsed[0], ["[first_name]", "[last_name]"])


class CacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous = {
            "use_cache": Config.use_cache,
            "save_cache": Config.save_cache,
            "cache_dir": Config.cache_dir,
            "perfect_split": Config.perfect_split,
            "perfect_directory": Config.perfect_directory,
            "perfect_cluster": Config.perfect_cluster,
            "data_directory": Config.data_directory,
            "num_examples": Config.num_examples,
        }

    def tearDown(self) -> None:
        for key, value in self.previous.items():
            setattr(Config, key, value)

    def test_cache_directory_is_resolved_at_call_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            Config.cache_dir = directory
            Config.save_cache = True
            Config.use_cache = False
            calls = []

            @cache_result()
            def operation(value: int) -> int:
                calls.append(value)
                return value * 2

            self.assertEqual(operation(3), 6)
            Config.save_cache = False
            Config.use_cache = True
            self.assertEqual(operation(3), 6)
            self.assertEqual(calls, [3])
            self.assertEqual(len(list(Path(directory).glob("*.pkl"))), 1)

    def test_disabled_cache_does_not_serialize_arguments(self):
        class Unserializable:
            def __reduce__(self):
                raise AssertionError("disabled cache must not serialize arguments")

        @cache_result()
        def identity(value):
            return value

        value = Unserializable()
        with patch.object(Config, "use_cache", False), patch.object(Config, "save_cache", False):
            self.assertIs(identity(value), value)

    def test_cache_version_recomputes_old_results_and_reuses_new_results(self):
        with tempfile.TemporaryDirectory() as directory:
            Config.cache_dir = directory
            Config.save_cache = True
            Config.use_cache = True
            calls = []

            def operation(value):
                calls.append(value)
                return len(calls)

            legacy = cache_result()(operation)
            updated = cache_result(version="trimmed-components-v1")(operation)
            self.assertEqual(legacy("value"), 1)
            self.assertEqual(updated("value"), 2)
            self.assertEqual(updated("value"), 2)
            self.assertEqual(legacy("value"), 1)
            self.assertEqual(calls, ["value", "value"])

    def test_perfect_split_maps_strings_to_nonsequential_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "Sample.csv").write_text("Alice,Jones\n", encoding="utf-8")
            Path(directory, "Sample_split.txt").write_text(
                "Jones\nlast_name: Jones\n[ITEM]\nAlice\nfirst_name: Alice\n",
                encoding="utf-8",
            )
            Config.data_directory = directory
            Config.num_examples = 1
            Config.perfect_directory = directory
            Config.perfect_split = True
            Config.use_cache = False

            @perfect_split_override()
            def value_split(query_dict, file):
                raise AssertionError("override was not used")

            @perfect_split_override_prompt()
            def label_split(query_dict, file):
                raise AssertionError("override was not used")

            query = {7: "Alice", 42: "Jones"}
            self.assertEqual(value_split(query, "Sample.csv")[42], {"last_name": "Jones"})
            self.assertEqual(label_split(query, "Sample.csv")[7], ["[first_name]"])

            Path(directory, "Full_split.txt").write_text(
                "0\nzero: 0\n[ITEM]\n1\none: 1\n[ITEM]\n2\ntwo: 2\n",
                encoding="utf-8",
            )
            Path(directory, "Full.csv").write_text("2,0\n", encoding="utf-8")
            self.assertEqual(value_split({2: "2"}, "Full.csv")[2], {"two": "2"})

    def test_perfect_clusters_are_restricted_to_loaded_target_examples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.joinpath("Sample.csv").write_text("a,A\nb,B\nc,C\n", encoding="utf-8")
            root.joinpath("Sample_cluster.txt").write_text(
                "a\nb\n[ITEM]\nc\n[ITEM]\nA\nB\nC\n", encoding="utf-8"
            )
            Config.data_directory = directory
            Config.perfect_directory = directory
            Config.num_examples = 1
            Config.perfect_cluster = True
            Config.use_cache = False

            @perfect_cluster_override()
            def cluster(file):
                raise AssertionError("override was not used")

            self.assertEqual(cluster("Sample.csv"), [{0}, {3}])


if __name__ == "__main__":
    unittest.main()
