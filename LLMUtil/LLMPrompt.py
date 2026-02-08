class LLMPrompt:

    BASE_SYS_SEMEXT = \
    """
    STRICT FORMAT RULES (MUST FOLLOW):
    - Output ONLY bracketed labels and the literal token [ITEM], one per line. No explanation, no extra text, no blank lines. Each bracketed label contains exactly one semantic component. Combining multiple components into one label, such as [first_name_and_last_name], is NOT allowed.
    - Split to smallest possible semantic components. DO NOT split on syllable or character level. The smallest semantic component cannot be igonred because of the original string. For example, if a name appears in an email address, first name and last name must still be identified regardless that it is an email.
    - Outputing a label that does not exist in the input is NOT allowed.
    - An abbreviation belonging to an entity must NOT be split from the entity, e.g. "Sample St." should return [street_name], not [street_name][street_abbreviation].
    - Preserve [ITEM] separators in the same positions as input.
    - Use lowercase underscore labels like [first_name], [house_number], if unsure, use[UNKNOWN].
    - Use world knowledge for ordering and split. E.g., "Iwata Satoru" -> [last_name][first_name]; month cannot be greater than 12.
    - Phone numbers are split every 3 to 4 digits, depending on the original string. If country code exists, separate it as [country_code], and the country code does not count towards the 3-4 digit grouping.

    You are a deterministic semantic splitter. Follow the STRICT FORMAT RULES above.
    Label choices: use specific semantic labels (e.g. [first_name], [first_three_digits], [year], [organization]). Use world knowledge for ordering and split. If unsure, use [UNKNOWN].
    Now process the following input (items separated by the literal token [ITEM]) — output only labels and [ITEM], one per line:


    """

    BASE_SYS_FURSP = \
    """
    STRICT FORMAT RULES (MUST FOLLOW):
    - Output ONLY bracketed labels and the literal token [ITEM], one per line. No explanation, no extra text, no blank lines. Each bracketed label contains exactly one semantic component. Combining multiple components into one label, such as [first_name_and_last_name], is NOT allowed.
    - Split to smallest possible semantic components. DO NOT split on syllable or character level. The smallest semantic component cannot be igonred because of the original string. For example, if a name appears in an email address, first name and last name must still be identified regardless that it is an email.
    - Outputing a label that does not exist in the input is NOT allowed.
    - An abbreviation belonging to an entity must NOT be split from the entity, e.g. "Sample St." should return [street_name], not [street_name][street_abbreviation].
    - Preserve [ITEM] separators in the same positions as input.
    - Use lowercase underscore labels like [first_name], [house_number], if unsure, use[UNKNOWN].
    - Use world knowledge for ordering and split. E.g., "Iwata Satoru" -> [last_name][first_name]; month cannot be greater than 12.
    - Phone numbers are split every 3 to 4 digits, depending on the original string. If country code exists, separate it as [country_code], and the country code does not count towards the 3-4 digit grouping.

    You are a deterministic semantic splitter. Follow the STRICT FORMAT RULES above.
    Label choices: use specific semantic labels (e.g. [first_name], [first_three_digits], [year], [organization]). Use world knowledge for ordering and split. If unsure, use [UNKNOWN].
    Ensure that at least the following semantic types are identified for each string: {sem_types}. Now process the following input (items separated by the literal token [ITEM]) — output only labels and [ITEM], one per line: 

    """

    BASE_SYS_FINALSP = \
    """
    STRICT FORMAT RULES (MUST FOLLOW):
    - Output ONLY bracketed labels and the literal token [ITEM], one per line. No explanation, no extra text, no blank lines. Each bracketed label contains exactly one semantic component. Combining multiple components into one label, such as [first_name_and_last_name], is NOT allowed.
    - Split to smallest possible semantic components. DO NOT split on syllable or character level. The smallest semantic component cannot be igonred because of the original string. For example, if a name appears in an email address, first name and last name must still be identified regardless that it is an email.
    - Outputing a label that does not exist in the input is NOT allowed.
    - An abbreviation belonging to an entity must NOT be split from the entity, e.g. "Sample St." should return [street_name], not [street_name][street_abbreviation].
    - Preserve [ITEM] separators in the same positions as input.
    - Use lowercase underscore labels like [first_name], [house_number], if unsure, use[UNKNOWN].
    - Use world knowledge for ordering and split. E.g., "Iwata Satoru" -> [last_name][first_name]; month cannot be greater than 12.
    - Phone numbers are split every 3 to 4 digits, depending on the original string. If country code exists, separate it as [country_code], and the country code does not count towards the 3-4 digit grouping.

    You are a deterministic semantic splitter. Follow the STRICT FORMAT RULES above.
    Label choices: use specific semantic labels (e.g. [first_name], [first_three_digits], [year], [organization]). Use world knowledge for ordering and split. If unsure, use [UNKNOWN].
    Ensure that at least the following semantic types are identified for each string: {sem_types}, unless they can be further split. Now process the following input (items separated by the literal token [ITEM]) — output only labels and [ITEM], one per line: 

    """

    OLLAMA_BASE = \
    """
    STRICT FORMAT RULES (MUST FOLLOW):
        - Output ONLY bracketed labels, one per line. No explanation, no extra text, no blank lines.
        - For each semantic component in the input item (ignore punctuation), output exactly one bracketed label in the same order.
        - Use exclusively the labels provided from label choices.
        - If unsure, use [UNKNOWN].
        - Use world knowledge for cultural ordering.
    You are a deterministic semantic splitter. Follow the STRICT FORMAT RULES above.
    For each semantic component, output a single bracketed label per line.
    Label choices: {sem_types}
    Examples:
    {examples}
    Now process the following input exactly as given:
    {query}
    """

    OLLAMA_ORIG = "Label the semantic parts in the order of occurrence in the text: {}. Use square brackets [] to enclose each part in your answer. If a part is not present, skip it. The response should contain **exclusively the labels mentioned above** without additional text. Use world knowledge to decide the order of occurrence, e.g. some names are more likely to be a family name; month cannot be greater than 12; The cultural ordering or a common format may indicate the order of the semantic components. The first two are examples. Only the third is what you need to answer.\n\n"