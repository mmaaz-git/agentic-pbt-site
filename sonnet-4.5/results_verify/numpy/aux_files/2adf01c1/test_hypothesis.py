import numpy as np
import numpy.char as char
from hypothesis import given, strategies as st

@given(st.lists(st.text(alphabet=st.characters(min_codepoint=0x100, max_codepoint=0x1000, blacklist_categories=('Cs',)), min_size=1, max_size=10), min_size=1, max_size=10))
def test_upper_lower_unicode(strings):
    arr = np.array(strings, dtype=str)
    upper_result = char.upper(arr)

    for i in range(len(strings)):
        assert upper_result[i] == strings[i].upper()

# Test with the specific failing input
if __name__ == "__main__":
    # First test the specific case
    print("Testing specific case: ['ß']")
    strings = ['ß']
    arr = np.array(strings, dtype=str)
    upper_result = char.upper(arr)

    print(f"Input: {strings[0]!r}")
    print(f"numpy.char.upper result: {upper_result[0]!r}")
    print(f"Python str.upper result: {strings[0].upper()!r}")

    try:
        assert upper_result[0] == strings[0].upper()
        print("✓ Assertion passed")
    except AssertionError:
        print("✗ Assertion failed: Results don't match")

    # Now run hypothesis testing
    print("\nRunning Hypothesis tests...")
    test_upper_lower_unicode()