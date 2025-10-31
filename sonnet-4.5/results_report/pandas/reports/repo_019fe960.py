import pandas.api.types as pat

# Test various invalid regex patterns that should return False
# but instead raise re.PatternError

test_cases = [
    ')',    # unbalanced parenthesis
    '?',    # nothing to repeat
    '*',    # nothing to repeat
    '+',    # nothing to repeat
    '(',    # missing closing parenthesis
    '[',    # unterminated character set
    '\\',   # bad escape at end
]

print("Testing pandas.api.types.is_re_compilable with invalid regex patterns:")
print("-" * 60)

for pattern in test_cases:
    print(f"\nInput: {repr(pattern)}")
    print("Expected: False")
    print("Actual: ", end="")
    try:
        result = pat.is_re_compilable(pattern)
        print(f"{result}")
    except Exception as e:
        print(f"Raised {e.__class__.__name__}: {e}")