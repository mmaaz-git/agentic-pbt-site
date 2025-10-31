import numpy as np
from hypothesis import given, strategies as st, settings


@given(st.text(min_size=1, max_size=50))
@settings(max_examples=1000)
def test_numpy_array_preserves_strings(s):
    arr = np.array([s])
    assert arr[0] == s, f"np.array should preserve all characters in strings. Input: {repr(s)}, Output: {repr(arr[0])}"

# Run the test
if __name__ == "__main__":
    import sys
    try:
        test_numpy_array_preserves_strings()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)