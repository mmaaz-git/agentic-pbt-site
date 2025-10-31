from hypothesis import given, strategies as st, settings
import numpy.char as char
import numpy as np

@given(st.lists(st.text(), min_size=1), st.text())
@settings(max_examples=500)
def test_find_matches_python(strings, substring):
    arr = np.array(strings)
    result = char.find(arr, substring)

    for original, found_idx in zip(arr, result):
        expected = original.find(substring)
        assert found_idx == expected, \
            f"find mismatch: '{original}'.find('{substring}') -> {found_idx} (expected {expected})"

if __name__ == "__main__":
    # Test with the specific failing input
    print("Testing with specific failing input: strings=[''], substring='\\x00'")
    strings = ['']
    substring = '\x00'
    arr = np.array(strings)
    result = char.find(arr, substring)
    expected = strings[0].find(substring)
    print(f"numpy result: {result[0]}")
    print(f"python result: {expected}")
    print(f"Match: {result[0] == expected}")

    # Run the full hypothesis test
    print("\nRunning full hypothesis test...")
    try:
        test_find_matches_python()
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")