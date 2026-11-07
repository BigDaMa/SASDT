from __future__ import annotations

from typing import Any, Dict

from Utils.config import Config
from Utils.IterPrintUtil import IterPrintUtil


class Evaluation:
    @staticmethod
    def evaluate(
        transformation: Dict[int, dict[str, Any]],
        ground_truth: Dict[int, str],
        original_strings: Dict[int, str] | None = None,
    ) -> dict[str, Any]:
        correct = 0
        incorrect_items = []
        output_items = []
        for value_id, expected in ground_truth.items():
            prediction_record = transformation.get(int(value_id))
            prediction = (
                str(prediction_record.get("transformation", "")).strip()
                if prediction_record
                else ""
            )
            expected_text = str(expected).strip()
            components = (
                prediction_record.get("original", {}) if prediction_record else {}
            )
            if not isinstance(components, dict):
                components = {}
            original_string = (
                prediction_record.get("orig_txt", "") if prediction_record else ""
            )
            if original_strings is not None:
                original_string = original_strings.get(int(value_id), original_string)
            output_items.append(
                {
                    "original string": str(original_string),
                    "sem. components": [
                        {str(semantic_type): semantic_component}
                        for semantic_type, semantic_component in components.items()
                    ],
                    "transformation": prediction,
                    "ground truth": expected_text,
                }
            )
            if prediction == expected_text:
                correct += 1
            else:
                incorrect_items.append(
                    {
                        "index": value_id,
                        "split": components,
                        "transformation": prediction,
                        "ground_truth": expected_text,
                    }
                )
        total = len(ground_truth)
        incorrect = len(incorrect_items)
        return {
            "accuracy": correct / total if total else 0.0,
            "correct": correct,
            "incorrect": incorrect,
            "incorrect_items": incorrect_items,
            "output_items": output_items,
        }

    @staticmethod
    def print_eval_result(eval_result: dict[str, Any]) -> None:
        if Config.verbose:
            print(f"Accuracy: {eval_result['accuracy']}")
            print(f"Correct: {eval_result['correct']}")
            print(f"Incorrect: {eval_result['incorrect']}")

        output_items = eval_result.get("output_items", [])
        correct_items = [
            item
            for item in output_items
            if item.get("transformation") == item.get("ground truth")
        ]
        incorrect_items = [
            item
            for item in output_items
            if item.get("transformation") != item.get("ground truth")
        ]
        print("Transformation results:")
        if Config.verbose:
            print("Correct:")
            IterPrintUtil.print(correct_items)
        print("Wrong:")
        IterPrintUtil.print(incorrect_items)
