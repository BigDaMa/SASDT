from typing import Dict

from Utils.config import Config
from Utils.IterPrintUtil import IterPrintUtil


class Evaluation:

    @staticmethod
    def evaluate(transformation: Dict[str, str], ground_truth: Dict[str, str]) -> dict:
        correct = 0
        incorrect = 0
        incorrect_items = []
        for key, val in transformation.items():
            idx = int(key)
            gt = ground_truth[idx].strip()
            if val['transformation'] == gt:
                correct += 1
            else:
                incorrect += 1
                incorrect_items.append({
                    "index": idx,
                    "split": val['original'],
                    "transformation": val["transformation"],
                    "ground_truth": gt
                })
        accuracy = float(correct) / float(correct + incorrect) if (correct + incorrect) > 0 else 0.0
        return {
            "accuracy": accuracy,
            "correct": correct,
            "incorrect": incorrect,
            "incorrect_items": incorrect_items
        }
    
    @staticmethod
    def print_eval_result(eval_result: dict):
        print(f"Accuracy: {eval_result['accuracy']}")
        print(f"Correct: {eval_result['correct']}")
        print(f"Incorrect: {eval_result['incorrect']}")
        if Config.verbose:
            print("Incorrect items:")
            IterPrintUtil.print(eval_result['incorrect_items'])