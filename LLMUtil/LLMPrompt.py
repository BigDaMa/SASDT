from collections.abc import Iterable
import re


class LLMPrompt:
    """Complete named templates for every provider and transformation stage."""

    ITEM_SEPARATOR = "\n[ITEM]\n"
    ORIGINAL_STRING_PREFIX = "ORIGINAL_STRING: "

    ORIGINAL_STRING_RESPONSE_FORMAT = """\
Each input begins with `ORIGINAL_STRING: <input>`. For every result, return that
exact line once followed by its split; do not repeat the input elsewhere. The header is
metadata, not a component.
"""

    GPT_SEM_COMPONENT = """\
Inputs are strings separated by [ITEM]. Split each into the smallest
real-world-meaningful components; never split non-numeric text into characters or
syllables. Choose the broadest semantic type that uniquely identifies the intended
substring in every applicable item; refine only to remove ambiguity. Use consistent
lowercase_underscore types. Return `type: exact substring` lines in any order:
exact, non-overlapping substrings only; omit absent types and number repeats
by their position in the original string. Return
one same-order block per input, separated by [ITEM].
{original_string_response_format}Output only those blocks.
"""

    GPT_SEM_COMPONENT_WITH_TYPES = """\
Inputs are strings separated by [ITEM]. Extract only these semantic types by
real-world meaning: {semantic_types}. Do not merge, split, or rename types.
Do not use the label that is not provided. Return
present types as `type: exact substring` lines in any order: exact,
non-overlapping substrings only; omit absent types and number repeats
by their position in the original string. Return one
same-order block per input, separated by [ITEM].
{original_string_response_format}Output only those blocks.
"""

    GPT_SEM_REFINEMENT = """\
Each [ITEM]-separated block is an existing `type: exact substring` split. Refine it
into the smallest components with distinct real-world meanings: {semantic_types}.
Never split non-numeric text into characters or syllables. Choose the broadest
semantic type that uniquely identifies the intended substring in every applicable
item; refine only to remove ambiguity. New type names may differ; preserve an
unrefinable line exactly. Use types consistently and omit absent types. Return one
same-order block per input, separated by [ITEM].
{original_string_response_format}Output only those blocks.
"""

    GPT_SEM_FINAL_COMPONENT = """\
Inputs are strings separated by [ITEM]. Split each into the smallest
real-world-meaningful components; never split non-numeric text into characters or
syllables. Choose the broadest semantic type that uniquely identifies the intended
substring in every applicable item; refine only to remove ambiguity. Use consistent
lowercase_underscore types. Return `type: exact substring` lines in any order:
exact, non-overlapping substrings only; omit absent types and number repeats
by their position in the original string. Use this
final semantic type vocabulary wherever its meanings occur: {semantic_types}. Return
one same-order block per input, separated by [ITEM].
{original_string_response_format}Output only those blocks.
"""

#     GPT_LABEL_COMPONENT = """\
# Inputs are strings separated by [ITEM]. Identify the types of the SMALLEST  real-world-meaningful components; never split non-numeric text into characters or
# syllables. The returned labels should not be able to be split further unless further splits causes the abbreviation to be separated from the original type or the labels cannot generalize to all the examples. Use the broadest label that uniquely identifies the component in every applicable item. Emit one lowercase_underscore label per component in source order; ignore punctuation. Use labels consistently, number repeats, and use [UNKNOWN] only when uncertain. Return one same-order block per input, separated by [ITEM]. Abbreviations that belong to a component should remain with that component.
# {original_string_response_format}Output only the requested blocks, with each label in square brackets. Note that the same substring often have the same label among different items.
# """

    GPT_LABEL_COMPONENT = """\
Inputs are string pairs, each [ITEM] block contains two related strings that must be analyzed jointly. Identify the types of the SMALLEST real-world-meaningful components; never split non-numeric text into characters or syllables. If a string consists of only digits after removing non-alphanumeric characters, and the lengths are generally similar, it can be split into groups of digits depending on the context. Use the correspondence between the two strings in each pair to choose component boundaries. If a meaningful component in one string appears inside a larger component in the other, split the larger component to expose the matching substring. Conceptually, each label identifier within a pair corresponds to one substring value. Components appearing only in one string are allowed; their absence from the other string does not itself require further splitting. Abbreviations are NOT ALLOWED to be separated from the original type. Emit one lowercase_underscore label per component in source order; ignore punctuation. Append a number ONLY when the same label is repeated in the same string. Use labels consistently, and use [UNKNOWN] only when uncertain. Never split further when it breaks the integrity of a real-world meaning. The level of split are often the same among all provided strings. Return one composition block per string in the provided order, giving two blocks per pair, with [ITEM] between every returned block. Return the compositions in the order of the provided strings.
{original_string_response_format}Output only the requested blocks containing semantic labels only, with each label in square brackets.
"""

#     GPT_LABEL_COMPONENT = """\
# Segment each string into the smallest spans with meaningful real-world semantics and assign each span a label in square brackets. The input consists of multiple [ITEM] blocks, each containing two related strings that must be analyzed jointly. Do not split by individual characters or syllables. Abbreviations are NOT ALLOWED to be separated from the original type. Skip substrings that serves as delimiters. Do not return the original substring. Use [UNKNOWN] when a span has no identifiable real-world meaning or its semantics cannot be determined reliably. If a string contains only digits after non-alphanumeric characters are removed, and have roughly equal lengths, you may use context or delimiters to divide it into fixed-length digit groups. Within each pair, the same substring must receive the same semantic label wherever it occurs in either string, and must not be absorbed into a larger span with a different label. Meaningful substrings occurring in only one string may still receive their own labels. Conceptually, each label identifier within a pair corresponds to one substring value. If the same semantic category applies to multiple distinct substrings, append numeric suffixes starting from 1. The same substring must retain the same numbered label across both strings regardless of occurrence order. Use a consistent label vocabulary and segmentation granularity across all input blocks. Preserve the original span order in each string. Return one composition block per string in the provided order, giving two blocks per pair, with [ITEM] between every returned block. Return the compositions in the order of the provided strings. 
# {original_string_response_format}Output only the requested blocks containing semantic labels only, with each label in square brackets. Do not include any other text, explanation, or commentary.
# """

#     GPT_LABEL_REFINEMENT = """\
# Each [ITEM]-separated block is an existing label split. Refine labels that combine
# distinct real-world meanings into the smallest meaningful types: {semantic_types}.
# Never split into characters or syllables; preserve an unrefinable label. Use the
# broadest label that uniquely identifies the component in every applicable item;
# refine only to remove ambiguity. Use labels consistently and omit absent types.
# Return one same-order block per input, separated by [ITEM].
# {original_string_response_format}Output only the requested blocks, with each label in square brackets.
# """

    GPT_LABEL_REFINEMENT = """\
Each [ITEM] block contains two corresponding strings, each followed by
its current composition. Refine their labels into the smallest real-world-meaningful
components, using these types where applicable: {semantic_types}. If a component
in either string appears inside a larger component in the other, split the larger
component and give the matching substrings the same label. Components found only
in one string are allowed; absence from the other string does not require splitting.
Never split non-numeric text into characters or syllables; keep abbreviations with
their components. Preserve labels that need no refinement. Use consistent
lowercase_underscore labels in source order, omit absent types, number repeated
labels within a string, and use [UNKNOWN] only when uncertain.
Return one composition block per string in the provided order, giving two blocks
per pair, with [ITEM] between every returned block.
{original_string_response_format}Output only the requested blocks, with each label in square brackets.
"""

#     GPT_LABEL_FINAL_COMPONENT = """\
# Inputs are strings separated by [ITEM]. Split each into the smallest distinct
# real-world-meaningful components; never split non-numeric text into characters or
# syllables. Use the broadest label that uniquely identifies the component in every
# applicable item; refine only to remove ambiguity. Emit one lowercase_underscore label
# per component in source order; ignore punctuation. Use labels consistently, omit
# absent components, keep entity abbreviations with their entity, number repeats, and
# use [UNKNOWN] only when uncertain. Use these final semantic types wherever their
# real-world meanings occur; skip a type when it is absent from an item:
# {semantic_types}
# Return one same-order block per input, separated by [ITEM].
# {original_string_response_format}Output only the requested blocks, with each label in square brackets.
# """

    GPT_LABEL_FINAL_COMPONENT = """\
Each [ITEM] block contains one string. Split each into the smallest distinct
real-world-meaningful components; never split non-numeric text into characters or
syllables. If a string consists of only digits after removing non-alphanumeric characters, and the lengths are generally similar, it can be split into groups of digits depending on the context. Preserve the supplied semantic types and their component boundaries;
do not merge components into a broader type. Emit one lowercase_underscore label
per component in source order; ignore punctuation. Use labels consistently, omit
absent components, keep entity abbreviations with their entity, number repeats, and
use [UNKNOWN] only when uncertain. Use these final semantic types wherever their
real-world meanings occur:
{semantic_types}
Return one composition block per string in the provided order, with [ITEM]
between every returned block.
{original_string_response_format}Output only the requested blocks, with each label in square brackets.
"""

    SLM_SEM_COMPONENT = """\
{input_format} Split each into the smallest
real-world-meaningful components; never split non-numeric text into characters or
syllables. Choose the broadest semantic type that uniquely identifies the intended
substring in every applicable item; use consistent lowercase_underscore types. Return `type: exact substring` lines with exact, non-overlapping substrings only; omit absent types and number repeats by their position in the original string.
{result_format}
{original_string_response_format}Output only the requested split. No code, Markdown, JSON, explanation, or comment.

Input:
{input_text}
"""

    SLM_SEM_COMPONENT_WITH_TYPES = """\
{input_format} Extract the exact substrings that match the real-world meaning of the following semantic types: {semantic_types}. Do not merge, split, or rename types. Do not use the label that is not provided. The exact substring must come from the original string. Return existing types and corresponding substrings as `type: exact substring`, one per line, with exact, non-overlapping substrings only; omit absent types and number repeats by their position in the original string.
{result_format}
{original_string_response_format}Output only the requested split: no code, Markdown,
JSON, explanation, or commentary.

Input:
{input_text}
"""

    SLM_SEM_REFINEMENT = """\
{input_format} Refine each existing `type: exact substring` split
into the smallest components with distinct real-world meanings: {semantic_types}.
Never split non-numeric text into characters or syllables. Choose the broadest
semantic type that uniquely identifies the intended substring in every applicable
item; refine only to remove ambiguity. New type names may differ; preserve an
unrefinable line exactly. Use types consistently and omit absent types.
{result_format}
{original_string_response_format}Output only the requested split: no code, Markdown,
JSON, explanation, or commentary.

Input:
{input_text}
"""

    SLM_SEM_FINAL_COMPONENT = """\
{input_format} Split each into the smallest
real-world-meaningful components; never split non-numeric text into characters or
syllables. Choose the broadest semantic type that uniquely identifies the intended
substring in every applicable item; refine only to remove ambiguity. Use consistent
lowercase_underscore types. Return `type: exact substring` lines in any order:
exact, non-overlapping substrings only; omit absent types and number repeats
by their position in the original string. Use this
final semantic type vocabulary wherever its meanings occur: {semantic_types}.
{result_format}
{original_string_response_format}Output only the requested split: no code, Markdown,
JSON, explanation, or commentary.

Input:
{input_text}
"""

#     SLM_CLUSTER_LABEL = """\
# Identify the smallest real-world-meaningful components, without splitting words
# into characters or syllables. Ignore punctuation. Use the broadest supplied labels
# that identify the components, in source order, and use [UNKNOWN] when unsure. Omit the provided labels if they are not present in the input. Use the provided examples to understand the expected output format, but do not return results for the examples themselves.
# {response_format}
#
# Label choices: {semantic_types}
# Examples:
# {examples}
#
# Input:
# {input_text}
# """

    SLM_CLUSTER_LABEL = """\
Identify the smallest real-world-meaningful components, without splitting words
into characters or syllables. Ignore punctuation. Use supplied labels in source
order, preserving the component boundaries illustrated by the examples; do not
merge their separate components into a broader type. Use [UNKNOWN] when unsure
and omit labels absent from the input. Follow the examples' composition format,
but return results only for the queried strings.
{response_format}

Label choices: {semantic_types}
Examples:
{examples}

Input:
{input_text}
"""

    SLM_BATCH_INPUT_FORMAT = "Inputs are strings separated by [ITEM]."
    SLM_SINGLE_INPUT_FORMAT = "There is one input string."
    SLM_BATCH_RESULT_FORMAT = "Return one same-order block per input, separated by [ITEM]."
    SLM_SINGLE_RESULT_FORMAT = "Return one result for this input."
    SLM_LABEL_EXAMPLE = "{input_text}: {composition}"

    GPT_DIRECT_TRANSFORM = (
        "Answer each question on one line with no additional information. "
        "Do not repeat the question or add labels."
    )
    DIRECT_TRANSFORM_EXAMPLE = "Q: {input_text} A: {output_text}\n"
    DIRECT_TRANSFORM_QUESTION = "Q: {input_text} A:"
    DIRECT_TRANSFORM_STOP = "Q:"
    DIAGNOSTIC = "Instructions:\n{instructions}\n\nInput:\n{input_text}"

    @classmethod
    def gpt_sem_component(cls, *, original_strings: bool) -> str:
        return cls.GPT_SEM_COMPONENT.format(
            original_string_response_format=cls._original_string_format(original_strings)
        )

    @classmethod
    def gpt_sem_component_with_types(
        cls, *, semantic_types: str, original_strings: bool
    ) -> str:
        return cls.GPT_SEM_COMPONENT_WITH_TYPES.format(
            semantic_types=semantic_types,
            original_string_response_format=cls._original_string_format(original_strings),
        )

    @classmethod
    def gpt_sem_refinement(cls, *, semantic_types: str, original_strings: bool) -> str:
        return cls.GPT_SEM_REFINEMENT.format(
            semantic_types=semantic_types,
            original_string_response_format=cls._original_string_format(original_strings),
        )

    @classmethod
    def gpt_sem_final_component(
        cls, *, semantic_types: str, original_strings: bool
    ) -> str:
        return cls.GPT_SEM_FINAL_COMPONENT.format(
            semantic_types=semantic_types,
            original_string_response_format=cls._original_string_format(original_strings),
        )

    @classmethod
    def gpt_label_component(cls, *, original_strings: bool) -> str:
        return cls.GPT_LABEL_COMPONENT.format(
            original_string_response_format=cls._original_string_format(original_strings)
        )

    @classmethod
    def gpt_label_refinement(cls, *, semantic_types: str, original_strings: bool) -> str:
        return cls.GPT_LABEL_REFINEMENT.format(
            semantic_types=semantic_types,
            original_string_response_format=cls._original_string_format(original_strings),
        )

    @classmethod
    def gpt_label_final_component(
        cls, *, semantic_types: str, original_strings: bool
    ) -> str:
        return cls.GPT_LABEL_FINAL_COMPONENT.format(
            semantic_types=semantic_types,
            original_string_response_format=cls._original_string_format(original_strings),
        )

    @classmethod
    def slm_sem_component(
        cls, *, input_text: str, original_strings: bool, batched: bool = False
    ) -> str:
        return cls.SLM_SEM_COMPONENT.format(
            input_text=input_text,
            original_string_response_format=cls._original_string_format(original_strings),
            **cls._slm_query_format(batched),
        )

    @classmethod
    def slm_sem_component_with_types(
        cls, *, input_text: str, semantic_types: str, original_strings: bool,
        batched: bool = False,
    ) -> str:
        return cls.SLM_SEM_COMPONENT_WITH_TYPES.format(
            input_text=input_text,
            semantic_types=cls._slm_type_choices(semantic_types),
            original_string_response_format=cls._original_string_format(original_strings),
            **cls._slm_query_format(batched),
        )

    @classmethod
    def slm_sem_refinement(
        cls, *, input_text: str, semantic_types: str, original_strings: bool,
        batched: bool = False,
    ) -> str:
        return cls.SLM_SEM_REFINEMENT.format(
            input_text=input_text,
            semantic_types=cls._slm_type_choices(semantic_types),
            original_string_response_format=cls._original_string_format(original_strings),
            **cls._slm_query_format(batched),
        )

    @classmethod
    def slm_sem_final_component(
        cls, *, input_text: str, semantic_types: str, original_strings: bool,
        batched: bool = False,
    ) -> str:
        return cls.SLM_SEM_FINAL_COMPONENT.format(
            input_text=input_text,
            semantic_types=cls._slm_type_choices(semantic_types),
            original_string_response_format=cls._original_string_format(original_strings),
            **cls._slm_query_format(batched),
        )

    @classmethod
    def slm_cluster_label(
        cls, *, input_text: str, semantic_types: str,
        examples: Iterable[tuple[str, str]],
        original_strings: bool = False, batched: bool = False,
    ) -> str:
        if batched:
            response_format = (
                "Return one composition block per string in the provided order, "
                "with its ORIGINAL_STRING header followed by one composition line "
                "and [ITEM] between every returned block."
                if original_strings else
                "Return one composition block per string in the provided order, "
                "with [ITEM] between every returned block."
            )
        else:
            response_format = "Return only the composition for the input below."
        return cls.SLM_CLUSTER_LABEL.format(
            input_text=input_text,
            semantic_types=cls._slm_type_choices(semantic_types),
            examples="\n".join(
                cls.SLM_LABEL_EXAMPLE.format(input_text=text, composition=composition)
                for text, composition in examples
            ),
            response_format=response_format,
        )

    @staticmethod
    def _slm_type_choices(semantic_types: str) -> str:
        """Omit the unknown placeholder from comma/conjunction-separated choices."""
        choices = re.split(r",\s*(?:and\s+)?|\s+and\s+", semantic_types, flags=re.IGNORECASE)
        known = [choice for choice in choices
                 if choice.strip().strip("[]").strip().casefold() != "unknown"]
        if len(known) == len(choices):
            return semantic_types
        return ", ".join(choice.strip() for choice in known if choice.strip())

    @classmethod
    def _slm_query_format(cls, batched: bool) -> dict[str, str]:
        return {
            "input_format": cls.SLM_BATCH_INPUT_FORMAT if batched else cls.SLM_SINGLE_INPUT_FORMAT,
            "result_format": cls.SLM_BATCH_RESULT_FORMAT if batched else cls.SLM_SINGLE_RESULT_FORMAT,
        }

    @classmethod
    def join_items(cls, items: Iterable[str]) -> str:
        return cls.ITEM_SEPARATOR.join(items)

    @classmethod
    def join_pairs(cls, items: list[str]) -> str:
        if len(items) % 2:
            raise ValueError("paired queries require two strings per pair")
        return cls.join_items("\n".join(items[index:index + 2])
                              for index in range(0, len(items), 2))

    @classmethod
    def composition_refinement_item(
        cls, *, original: str, composition: str, original_strings: bool,
    ) -> str:
        body = f"Current composition: {composition}"
        return body if original_strings else f"{original}\n{body}"

    @classmethod
    def original_string_item(cls, *, original: str, body: str) -> str:
        header = f"{cls.ORIGINAL_STRING_PREFIX}{original}"
        return header if body == original else f"{header}\n{body}"

    @classmethod
    def diagnostic(cls, *, instructions: str, input_text: str) -> str:
        # SLM templates already include the input heading. Preserve that rendered
        # template when a mocked/custom provider cannot capture its actual prompt.
        if instructions.rstrip().endswith("\nInput:"):
            return f"{instructions.rstrip()}\n{input_text}\n"
        return cls.DIAGNOSTIC.format(instructions=instructions, input_text=input_text)

    @classmethod
    def with_original_string_response_format(cls, instructions: str) -> str:
        """Apply the complete header-mode response format to a custom instruction."""

        return f"{instructions.rstrip()}\n{cls.ORIGINAL_STRING_RESPONSE_FORMAT}"

    @classmethod
    def direct_transform_examples(cls, examples: list[dict[str, str]]) -> str:
        return "".join(
            cls.DIRECT_TRANSFORM_EXAMPLE.format(
                input_text=example["input"],
                output_text=example["output"],
            )
            for example in examples
        )

    @classmethod
    def direct_transform_questions(cls, questions: list[str]) -> str:
        return "\n".join(
            cls.DIRECT_TRANSFORM_QUESTION.format(input_text=question)
            for question in questions
        )

    @classmethod
    def direct_transform_question(
        cls, *, examples: list[dict[str, str]], question: str
    ) -> str:
        return (
            f"{cls.GPT_DIRECT_TRANSFORM}\n"
            f"{cls.direct_transform_examples(examples)}"
            f"{cls.DIRECT_TRANSFORM_QUESTION.format(input_text=question)}"
        )

    @classmethod
    def _original_string_format(cls, enabled: bool) -> str:
        return cls.ORIGINAL_STRING_RESPONSE_FORMAT if enabled else ""
