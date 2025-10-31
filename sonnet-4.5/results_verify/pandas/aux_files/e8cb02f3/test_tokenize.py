import tokenize
from io import StringIO

# Test what happens when tokenizing a string with null byte
test_string = "`\x00`"
print(f"Testing tokenize with: {test_string!r}")

try:
    line_reader = StringIO(test_string).readline
    token_generator = tokenize.generate_tokens(line_reader)
    for tok in token_generator:
        print(f"Token: {tok}")
except tokenize.TokenError as e:
    print(f"TokenError raised: {e}")
except SyntaxError as e:
    print(f"SyntaxError raised: {e}")

# Test what Python's tokenizer does with null bytes in general
print("\nTesting Python tokenizer with null byte:")
try:
    exec('\x00')
except SyntaxError as e:
    print(f"SyntaxError from exec: {e}")
except Exception as e:
    print(f"Other exception from exec: {type(e).__name__}: {e}")