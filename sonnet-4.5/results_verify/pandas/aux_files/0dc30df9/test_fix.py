import tokenize
from io import StringIO
from typing import Iterator, Hashable

BACKTICK_QUOTED_STRING = 100

def tokenize_backtick_quoted_string(
    token_generator: Iterator[tokenize.TokenInfo], source: str, string_start: int
) -> tuple[int, str]:
    for _, tokval, start, _, _ in token_generator:
        if tokval == "`":
            string_end = start[1]
            break
    return BACKTICK_QUOTED_STRING, source[string_start:string_end]

def tokenize_string(source: str) -> Iterator[tuple[int, str]]:
    line_reader = StringIO(source).readline
    token_generator = tokenize.generate_tokens(line_reader)

    for toknum, tokval, start, _, _ in token_generator:
        if tokval == "`":
            try:
                yield tokenize_backtick_quoted_string(
                    token_generator, source, string_start=start[1] + 1
                )
            except Exception as err:
                raise SyntaxError(f"Failed to parse backticks in '{source}'.") from err
        else:
            yield toknum, tokval

def create_valid_python_identifier(name: str) -> str:
    # Simplified version for testing
    return name

def clean_column_name_fixed(name: Hashable) -> Hashable:
    """Fixed version that catches both SyntaxError and TokenError"""
    try:
        tokenized = tokenize_string(f"`{name}`")
        tokval = next(tokenized)[1]
        return create_valid_python_identifier(tokval)
    except (SyntaxError, tokenize.TokenError):
        return name

# Test the fixed version
test_cases = [
    '\x00',  # null byte
    'valid_name',
    'name with spaces',
    '123invalid',
    '\x00embedded\x00nulls\x00'
]

for test in test_cases:
    print(f"Testing: {repr(test)}")
    try:
        result = clean_column_name_fixed(test)
        print(f"  Result: {repr(result)}")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")