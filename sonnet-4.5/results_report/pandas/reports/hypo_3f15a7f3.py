from hypothesis import given, strategies as st
import numpy as np
from pandas.core import algorithms as alg

@given(st.lists(st.text(min_size=0, max_size=10)))
def test_factorize_round_trip(values):
    if len(values) == 0:
        return

    arr = np.array(values)
    codes, uniques = alg.factorize(arr)

    reconstructed = uniques.take(codes[codes >= 0])
    original_without_na = arr[codes >= 0]

    if len(reconstructed) > 0 and len(original_without_na) > 0:
        assert all(a == b for a, b in zip(reconstructed, original_without_na))

if __name__ == "__main__":
    # Run the property test
    try:
        test_factorize_round_trip()
        print("All tests passed!")
    except AssertionError:
        print("Test failed!")
        import traceback
        traceback.print_exc()

    # Test the specific failing case directly
    print("\nTesting specific failing case: ['', '\\x000']")
    values = ['', '\x000']
    arr = np.array(values)
    codes, uniques = alg.factorize(arr)
    reconstructed = uniques.take(codes[codes >= 0])
    original_without_na = arr[codes >= 0]
    try:
        assert all(a == b for a, b in zip(reconstructed, original_without_na))
        print("Specific test passed!")
    except AssertionError:
        print("Specific test failed: reconstructed values don't match originals")
        print(f"  Original: {list(original_without_na)}")
        print(f"  Reconstructed: {list(reconstructed)}")