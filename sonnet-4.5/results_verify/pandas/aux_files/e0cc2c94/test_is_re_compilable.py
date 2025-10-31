import re
from hypothesis import given, strategies as st
import pandas.api.types as pat


@given(st.text())
def test_is_re_compilable_for_strings(x):
    result = pat.is_re_compilable(x)
    if result:
        try:
            re.compile(x)
        except re.error:
            raise AssertionError(f"is_re_compilable({x!r}) returned True but re.compile raised error")

# Run the hypothesis test
if __name__ == "__main__":
    # Test function without hypothesis decorator for specific inputs
    def test_specific_input(x):
        result = pat.is_re_compilable(x)
        if result:
            try:
                re.compile(x)
            except re.error:
                raise AssertionError(f"is_re_compilable({x!r}) returned True but re.compile raised error")

    # Test with the specific failing input mentioned
    print("Testing with '[' ...")
    try:
        test_specific_input('[')
        print("Test passed with '['")
    except Exception as e:
        print(f"Test failed with '[': {type(e).__name__}: {e}")

    # Test with other invalid regex patterns mentioned
    print("\nTesting with '?' ...")
    try:
        test_specific_input('?')
        print("Test passed with '?'")
    except Exception as e:
        print(f"Test failed with '?': {type(e).__name__}: {e}")

    print("\nTesting with '(unclosed' ...")
    try:
        test_specific_input('(unclosed')
        print("Test passed with '(unclosed'")
    except Exception as e:
        print(f"Test failed with '(unclosed': {type(e).__name__}: {e}")