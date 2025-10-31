from hypothesis import given, strategies as st, assume
import numpy.ma as ma
import numpy as np


@given(
    m1=st.lists(st.booleans(), min_size=1, max_size=50),
    m2=st.lists(st.booleans(), min_size=1, max_size=50)
)
def test_mask_or_symmetry(m1, m2):
    assume(len(m1) == len(m2))

    result1 = ma.mask_or(m1, m2)
    result2 = ma.mask_or(m2, m1)

    if result1 is ma.nomask and result2 is ma.nomask:
        pass
    elif result1 is ma.nomask or result2 is ma.nomask:
        assert False, f"mask_or should be symmetric, but one is nomask: {result1} vs {result2}"
    else:
        assert np.array_equal(result1, result2), f"mask_or not symmetric: {result1} vs {result2}"

if __name__ == "__main__":
    print("Running hypothesis test with example input...")
    try:
        # Manually test with the failing input
        m1 = [False]
        m2 = [False]
        print(f"Testing with m1={m1}, m2={m2}")

        result1 = ma.mask_or(m1, m2)
        result2 = ma.mask_or(m2, m1)

        if result1 is ma.nomask and result2 is ma.nomask:
            print("Both results are nomask")
        elif result1 is ma.nomask or result2 is ma.nomask:
            print(f"ERROR: mask_or not symmetric, one is nomask: {result1} vs {result2}")
        else:
            if np.array_equal(result1, result2):
                print(f"Test passed! Results are symmetric: {result1}")
            else:
                print(f"ERROR: mask_or not symmetric: {result1} vs {result2}")
    except Exception as e:
        print(f"Test failed with exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()