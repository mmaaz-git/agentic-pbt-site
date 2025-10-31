from hypothesis import given, strategies as st
from dask.utils import format_bytes

@given(st.integers(min_value=0, max_value=2**60))
def test_format_bytes_length_bound(n):
    result = format_bytes(n)
    print(f"Testing n={n}, result='{result}', len={len(result)}")
    assert len(result) <= 10, f"Failed for n={n}, result='{result}', len={len(result)}"

# Run the test
if __name__ == "__main__":
    # Test with the specific failing value mentioned in the bug report
    n = 1_125_894_277_343_089_729
    result = format_bytes(n)
    print(f"\nSpecific test with n={n}")
    print(f"Result: '{result}'")
    print(f"Length: {len(result)}")
    print(f"n < 2**60: {n < 2**60}")

    # Now run the hypothesis test
    print("\nRunning hypothesis test...")
    try:
        test_format_bytes_length_bound()
    except AssertionError as e:
        print(f"Hypothesis test failed: {e}")