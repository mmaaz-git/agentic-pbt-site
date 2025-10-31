from hypothesis import given, strategies as st, example
import pandas.util


@given(st.text())
@example('ß')
@example('ﬁle')
@example('ﬂow')
def test_capitalize_preserves_length(s):
    result = pandas.util.capitalize_first_letter(s)
    if len(result) != len(s):
        print(f"FAILURE: '{s}' (len={len(s)}) -> '{result}' (len={len(result)})")
        return False
    return True

# Run the test
if __name__ == "__main__":
    import sys
    try:
        test_capitalize_preserves_length()
        print("Test passed!")
    except Exception as e:
        print(f"Test failed with error: {e}")
        sys.exit(1)