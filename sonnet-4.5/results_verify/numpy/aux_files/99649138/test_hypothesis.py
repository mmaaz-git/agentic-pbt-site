import numpy.ma as ma
from hypothesis import given, settings, strategies as st, assume


@settings(max_examples=1000)
@given(st.lists(st.booleans(), min_size=1, max_size=20))
def test_mask_or_accepts_lists(mask1):
    assume(len(mask1) > 0)
    mask2 = [not m for m in mask1]
    result = ma.mask_or(mask1, mask2)
    print(f"Success with mask1={mask1}")

if __name__ == "__main__":
    # Test with the failing input reported
    mask1 = [False]
    mask2 = [True]
    try:
        result = ma.mask_or(mask1, mask2)
        print(f"Test passed with mask1={mask1}, mask2={mask2}, result={result}")
    except Exception as e:
        print(f"Test failed with mask1={mask1}, mask2={mask2}")
        print(f"Error: {type(e).__name__}: {e}")

    # Run hypothesis tests
    try:
        test_mask_or_accepts_lists()
        print("All hypothesis tests passed")
    except Exception as e:
        print(f"Hypothesis test failed: {e}")