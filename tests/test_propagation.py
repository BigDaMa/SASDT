from __future__ import annotations

from collections import Counter
import itertools
import random
import unittest

from Propagate.propagate import Propagate
from Propagate.position import AbsPosPosition


def can_place_in_any_order(original, values):
    for ordered in set(itertools.permutations(value for value in values if value)):
        cursor = 0
        for value in ordered:
            start = original.find(value, cursor)
            if start < 0:
                break
            cursor = start + len(value)
        else:
            return True
    return False


def exhaustive_score(original, candidates):
    """Independent brute-force check of unordered, non-overlapping selection."""
    types = list(candidates)
    options = [[value for value, _ in counter.most_common()] or [""]
               for counter in candidates.values()]
    best_score = None
    for values in itertools.product(*options):
        if not can_place_in_any_order(original, values):
            continue
        score = sum(candidates[key][value] for key, value in zip(types, values))
        if best_score is None or score > best_score:
            best_score = score
    return best_score


class PropagationSelectionTests(unittest.TestCase):
    def test_trimmed_candidates_combine_support_and_preserve_internal_spaces(self):
        original = "  $Alice  Smith!  "
        candidates = Propagate._extract_candidates(original, [
            (AbsPosPosition(0), AbsPosPosition(None)),
            (AbsPosPosition(1), AbsPosPosition(-1)),
            (AbsPosPosition(2), AbsPosPosition(-2)),
            (AbsPosPosition(0), AbsPosPosition(2)),  # Only spaces.
            (AbsPosPosition(4), AbsPosPosition(-2)),  # Invalid cut inside "Alice".
        ])
        self.assertEqual(candidates, Counter({"$Alice  Smith!": 3}))

    def test_punctuation_does_not_discard_the_best_supported_valid_value(self):
        candidates = Propagate._extract_candidates("$12!", [
            (AbsPosPosition(0), AbsPosPosition(3)),
            (AbsPosPosition(0), AbsPosPosition(-1)),
            (AbsPosPosition(1), AbsPosPosition(3)),
            (AbsPosPosition(2), AbsPosPosition(3)),  # Invalid cut inside "12".
        ])
        self.assertEqual(candidates, Counter({"$12": 2, "12": 1}))
        self.assertEqual(Propagate._select_candidates("$12!", {"amount": candidates}),
                         {"amount": "$12"})

    def test_selection_matches_exhaustive_search_including_ties_and_missing_values(self):
        randomizer = random.Random(42)
        for case in range(300):
            original = "".join(randomizer.choices("abc", k=12))
            candidates = {}
            for index in range(randomizer.randrange(6)):
                values = randomizer.sample(["", "a", "b", "c", "ab", "bc", "ca", "missing"],
                                           randomizer.randrange(5))
                candidates[str(index)] = Counter({value: randomizer.randint(1, 4)
                                                  for value in values})
            with self.subTest(case=case, original=original, candidates=candidates):
                selected = Propagate._select_candidates(original, candidates)
                best = exhaustive_score(original, candidates)
                if best is not None:
                    self.assertTrue(can_place_in_any_order(original, selected.values()))
                    self.assertEqual(sum(candidates[key][value] for key, value in selected.items()),
                                     best)
                reversed_candidates = {
                    key: Counter(dict(reversed(list(counter.items()))))
                    for key, counter in reversed(list(candidates.items()))
                }
                self.assertEqual(Propagate._select_candidates(original, reversed_candidates), selected)

    def test_split_type_order_does_not_force_overlapping_or_lower_support_values(self):
        candidates = {"last": Counter({"Brown": 5, "Bob": 1}),
                      "first": Counter({"Bob": 5, "Brown": 1})}
        self.assertEqual(Propagate._select_candidates("Bob Brown", candidates),
                         {"first": "Bob", "last": "Brown"})
        self.assertTrue(Propagate.can_assemble_without_overlap("A A", ["A", "A"]))
        self.assertFalse(Propagate.can_assemble_without_overlap("A", ["A", "A"]))
        self.assertTrue(Propagate.can_assemble_without_overlap("New York / York", ["York", "New York"]))

    def test_many_alternatives_select_highest_support_without_overlap(self):
        candidates = {
            str(index): Counter({f"{index}:{option}": option + 1 for option in range(6)})
            for index in range(12)
        }
        original = " ".join(value for counter in candidates.values() for value in counter)
        self.assertEqual(Propagate._select_candidates(original, candidates),
                         {str(index): f"{index}:5" for index in range(12)})


if __name__ == "__main__":
    unittest.main()
