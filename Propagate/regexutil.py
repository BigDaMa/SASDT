class RegExUtil:
    regexes_char = [
        r"\b\d+\b", # integer
        r"\b[A-Za-z]+\b", # word
        r"\b[A-Z]+\b", # uppercase word
        r"\b[a-z]+\b", # lowercase word
        r'\b[A-Z][a-z]*([A-Z][a-z]*)+\b' # CamelCase word
    ]
    regexes_spec = [
        r"\s", # whitespace
        r"\t", # tab
        r",", # comma
        r"\.", # period
        r";", # semicolon
        r"!", # exclamation
        r"\(", # left parenthesis
        r"\)", # right parenthesis
        r"\[", # left bracket
        r"\]", # right bracket
        r"\"", # double quote
        r"\'", # single quote
        r"/", # forward slash
        r"\\", # backslash
        r"-", # hyphen
        r"\*", # asterisk
        r"\+", # plus
        r"_", # underscore
        r"=", # equals
        r"<", # less than
        r">", # greater than
        r"\{", # left curly brace
        r"\}", # right curly brace
        r"\|", # pipe
        r":", # colon
        r"\^", # caret
        r"&", # ampersand
        r"%", # percent
        r"\?", # question mark
        r"\$", # dollar sign
        r"@", # at sign
        r"~", # tilde
        r"`" # backtick
    ]

    regexes = regexes_char + regexes_spec