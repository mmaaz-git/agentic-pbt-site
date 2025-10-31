import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st, assume
from hypothesis.extra import numpy as npst

@given(npst.arrays(dtype=npst.integer_dtypes(), shape=npst.array_shapes()),
       st.data())
def test_unique_treats_masked_as_equal(arr, data):
    assume(arr.size > 1)
    mask = data.draw(npst.arrays(dtype=np.bool_, shape=arr.shape))
    assume(np.sum(mask) >= 2)

    marr = ma.array(arr, mask=mask)

    unique_result = ma.unique(marr)

    masked_in_result = ma.getmaskarray(unique_result)
    assert np.sum(masked_in_result) <= 1, f"Expected at most 1 masked value, but got {np.sum(masked_in_result)}"

# Test with the specific failing input
def test_specific_case():
    arr = np.array([32767, 32767, 32767], dtype=np.int16)
    mask = np.array([True, False, True])
    marr = ma.array(arr, mask=mask)

    unique_result = ma.unique(marr)
    masked_in_result = ma.getmaskarray(unique_result)
    num_masked = np.sum(masked_in_result)

    print(f"Array: {arr}")
    print(f"Mask: {mask}")
    print(f"Masked array: {marr}")
    print(f"Unique result: {unique_result}")
    print(f"Unique result mask: {masked_in_result}")
    print(f"Number of masked values in result: {num_masked}")

    assert num_masked <= 1, f"Expected at most 1 masked value, but got {num_masked}"

if __name__ == "__main__":
    # Test the specific case first
    print("Testing specific case...")
    try:
        test_specific_case()
        print("Specific test passed!")
    except AssertionError as e:
        print(f"Specific test failed: {e}")

    # Run hypothesis test
    print("\nRunning hypothesis tests...")
    test_unique_treats_masked_as_equal()