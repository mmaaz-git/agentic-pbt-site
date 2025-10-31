import numpy as np
import numpy.rec
from hypothesis import given, strategies as st

@given(st.lists(st.integers(), min_size=1, max_size=10))
def test_fromarrays_dtype_preserves_data(data):
    arr1 = np.array(data)
    arr2 = np.array(data)
    dtype = np.dtype([('a', 'i8'), ('b', 'i8')])
    rec = numpy.rec.fromarrays([arr1, arr2], dtype=dtype)
    assert np.array_equal(rec.a, arr1)

# Test with the specific failing input
def test_specific_failing_case():
    data = [9_223_372_036_854_775_808]
    arr1 = np.array(data)
    arr2 = np.array(data)
    dtype = np.dtype([('a', 'i8'), ('b', 'i8')])
    rec = numpy.rec.fromarrays([arr1, arr2], dtype=dtype)
    print(f"Original arr1: {arr1}")
    print(f"Original arr1 dtype: {arr1.dtype}")
    print(f"rec.a: {rec.a}")
    print(f"rec.a dtype: {rec.a.dtype}")
    print(f"Arrays equal: {np.array_equal(rec.a, arr1)}")

if __name__ == "__main__":
    # Run the specific failing case
    print("Testing specific failing case:")
    test_specific_failing_case()

    # Try to run hypothesis test
    print("\nRunning hypothesis test:")
    try:
        test_fromarrays_dtype_preserves_data()
    except Exception as e:
        print(f"Hypothesis test failed: {e}")