from hypothesis import given, strategies as st
import pandas.api.types as pat

@given(st.text())
def test_is_re_compilable(text):
    result = pat.is_re_compilable(text)
    if result:
        import re
        try:
            re.compile(text)
        except Exception as e:
            raise AssertionError(f"is_re_compilable returned True but re.compile failed: {e}")

if __name__ == "__main__":
    # Test with the specific failing input
    print("Testing with '[' ...")
    try:
        result = pat.is_re_compilable('[')
        print(f"Result: {result}")
    except Exception as e:
        print(f"Exception raised: {type(e).__name__}: {e}")

    # Test other invalid patterns
    invalid_patterns = ['(', ')', '?', '*', '+', '[abc', '(abc', '*abc']
    for pattern in invalid_patterns:
        try:
            result = pat.is_re_compilable(pattern)
            print(f"Pattern '{pattern}': {result}")
        except Exception as e:
            print(f"Pattern '{pattern}' raised: {type(e).__name__}: {e}")