from hypothesis import given, strategies as st, settings
from xarray.core.utils import is_uniform_spaced
import numpy as np

@given(n=st.integers(min_value=0, max_value=100))
@settings(max_examples=300)
def test_linspace_always_uniform(n):
    arr = np.linspace(0, 10, n)
    result = is_uniform_spaced(arr)
    assert result == True, f"linspace with {n} points should be uniformly spaced"

@given(size=st.integers(min_value=0, max_value=2))
@settings(max_examples=100)
def test_small_arrays_dont_crash(size):
    arr = list(range(size))
    result = is_uniform_spaced(arr)
    assert isinstance(result, bool)

if __name__ == "__main__":
    import traceback

    # Run the tests and catch the first failure details
    print("Running test_linspace_always_uniform...")
    try:
        test_linspace_always_uniform()
        print("test_linspace_always_uniform: PASSED")
    except Exception as e:
        print(f"test_linspace_always_uniform: FAILED")
        print(f"First failure with n=0 (empty array)")
        print(f"Error: {type(e).__name__}: {e}")
        print("\nTraceback:")
        traceback.print_exc()

    print("\n" + "="*50 + "\n")

    print("Running test_small_arrays_dont_crash...")
    try:
        test_small_arrays_dont_crash()
        print("test_small_arrays_dont_crash: PASSED")
    except Exception as e:
        print(f"test_small_arrays_dont_crash: FAILED")
        print(f"First failure with size=0 (empty array)")
        print(f"Error: {type(e).__name__}: {e}")
        print("\nTraceback:")
        traceback.print_exc()