from hypothesis import given, strategies as st, settings
import numpy as np

@given(st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=50))
@settings(max_examples=500)
def test_char_array_length_consistency(strings):
    arr = np.array(strings, dtype=str)

    for i, s in enumerate(strings):
        assert arr[i] == s, f"string array doesn't preserve strings. Original: {repr(s)}, Stored: {repr(arr[i])}"

# Test with the specific failing input mentioned
def test_specific_failing_case():
    strings = ['\x00']
    arr = np.array(strings, dtype=str)
    print(f"Input: {repr(strings[0])}, Output: {repr(arr[0])}")
    assert arr[0] == strings[0], f"Failed on null character"

if __name__ == "__main__":
    # First test the specific failing case
    print("Testing specific failing case:")
    try:
        test_specific_failing_case()
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")

    # Then run the property-based test
    print("\nRunning property-based test:")
    try:
        test_char_array_length_consistency()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")