from typing import Iterable

class IterPrintUtil:

    @staticmethod
    def print(iterable_to_print: Iterable, depth: int = 0):
        indent = "" if depth == 0 else "\t" * depth
        indent_content = "\t" * (depth + 1)
        if not isinstance(iterable_to_print, Iterable) or isinstance(iterable_to_print, tuple):
            print(f"{indent}{iterable_to_print}")
            return
        if isinstance(iterable_to_print, dict):
            print(f"{indent}{{")
            for key, value in iterable_to_print.items():
                print(f"{indent_content}{key}: ", end="")
                IterPrintUtil.print(value, depth + 1)
            print(f"{indent}}}")
        elif isinstance(iterable_to_print, list) or isinstance(iterable_to_print, set):
            bracket_open = "[" if isinstance(iterable_to_print, list) else "{"
            bracket_close = "]" if isinstance(iterable_to_print, list) else "}"
            print(f"{bracket_open}")
            for item in iterable_to_print:
                IterPrintUtil.print(item, depth + 1)
            print(f"{indent}{bracket_close}")
        else:
            print(f"{iterable_to_print}")
        return