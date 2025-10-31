from pandas.core.computation.parsing import tokenize_string
import tokenize

# Test various inputs to tokenize_string
test_cases = [
    "`valid`",
    "`with space`",
    "`with-dash`",
    "`123`",
    "`#comment`",  # This should raise SyntaxError according to comments
    "`\x00`",  # Null byte
]

for test in test_cases:
    print(f"\nTesting: {test!r}")
    try:
        result = list(tokenize_string(test))
        print(f"  Success: {result}")
    except SyntaxError as e:
        print(f"  SyntaxError: {e}")
    except tokenize.TokenError as e:
        print(f"  TokenError: {e}")
    except Exception as e:
        print(f"  Other: {type(e).__name__}: {e}")