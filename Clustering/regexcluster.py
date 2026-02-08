import pandas as pd
from grex import RegExpBuilder
from typing import List

class RegexProcessor:
    def __init__(self, r: str):
        self.r = r

    def getregex(self) -> str:
        """
        Returns the regex pattern as a string.
        """
        return self.r
    
    def __str__(self) -> str:
        """
        Returns the regex pattern as a string.
        """
        return self.r

    def regex_clean(self) -> str:
        r = self.r
        result = []
        i = 0
        n = len(r)
        bracket_chars = {'(', ')', '[', ']', '{', '}'}
        while i < n:
            if r[i] in bracket_chars:
                result.append(r[i])
                i += 1
                continue
            if r[i] == '\\' and i + 1 < n:
                seq = r[i:i+2]
                if i + 2 < n and r[i+2] == '{':
                    result.append(seq)
                    i += 2
                    continue
                count = 1
                j = i + 2
                while j + 1 < n and r[j] == '\\' and r[j+1] == seq[1]:
                    if j + 2 < n and r[j+2] == '{':
                        break
                    count += 1
                    j += 2
                if count > 1:
                    result.append(f'{seq}{{{count}}}')
                    i += 2 * count
                else:
                    result.append(seq)
                    i += 2
            else:
                if i + 1 < n and r[i+1] == '{':
                    result.append(r[i])
                    i += 1
                    continue
                count = 1
                j = i + 1
                while j < n and r[j] == r[i] and (j + 1 >= n or r[j+1] != '{') and r[j] not in bracket_chars:
                    count += 1
                    j += 1
                if count > 1:
                    result.append(f'{r[i]}{{{count}}}')
                    i += count
                else:
                    result.append(r[i])
                    i += 1
        return RegexProcessor(''.join(result))
    
    def expand(self) -> str:
        def parse(s):
            stack = []
            res = ['']
            i = 0
            while i < len(s):
                if s[i] == '(':  # handle parenthesis
                    count = 1
                    for j in range(i+1, len(s)):
                        if s[j] == '(': count += 1
                        elif s[j] == ')': count -= 1
                        if count == 0: break
                    inner = s[i+1:j]
                    if '|' in inner:
                        sub = parse(inner)
                        temp = []
                        for prefix in res:
                            for option in sub:
                                temp.append(prefix + option)
                        res = temp
                    else:
                        for k in range(len(res)):
                            res[k] += '(' + inner + ')'
                    i = j + 1
                elif s[i] == '|':
                    stack.append(res)
                    res = ['']
                    i += 1
                else:
                    for k in range(len(res)):
                        res[k] += s[i]
                    i += 1
            if stack:
                stack.append(res)
                res = []
                for group in stack:
                    res.extend(group)
            return res

        return parse(self.r)


def build_regex(data: List[str]) -> str:
    """
    Build a regex pattern from a list of strings.
    
    :param data: List of strings to build the regex from.
    :return: A regex pattern as a string.
    """
    pattern = (RegExpBuilder.from_test_cases(data)) \
            .with_conversion_of_digits() \
            .with_conversion_of_whitespace() \
            .with_conversion_of_words() \
            .with_conversion_of_repetitions() \
            .with_minimum_substring_length(1) \
            .with_capturing_groups() \
            .build()
    
    return pattern



df = pd.read_csv('data/Date.csv')
data = df.iloc[:5, 0].dropna().astype(str).tolist()
regex_pattern = build_regex(data)
regex_pattern = RegexProcessor(regex_pattern).regex_clean().expand()
