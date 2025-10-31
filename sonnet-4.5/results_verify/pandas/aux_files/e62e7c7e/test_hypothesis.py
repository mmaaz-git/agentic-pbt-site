import numpy as np
import pandas.core.algorithms as algorithms
from hypothesis import given, example, settings
from hypothesis.extra import numpy as npst

@given(npst.arrays(dtype=npst.integer_dtypes() | npst.floating_dtypes(),
                   shape=npst.array_shapes(max_dims=1)))
@example(np.array([0], dtype='>i8'))  # The specific failing case
@settings(max_examples=20)
def test_factorize_round_trip(arr):
    try:
        codes, uniques = algorithms.factorize(arr)
        assert len(codes) == len(arr)
        print(f"✓ Success for dtype: {arr.dtype}, shape: {arr.shape}")
        return True
    except ValueError as e:
        if "Big-endian buffer not supported" in str(e):
            print(f"✗ Failed for dtype: {arr.dtype}, shape: {arr.shape} - Big-endian error")
            raise
        else:
            raise

# Run the test
if __name__ == "__main__":
    print("Running property-based test:")
    print("="*50)
    try:
        test_factorize_round_trip()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTest failed with error: {e}")