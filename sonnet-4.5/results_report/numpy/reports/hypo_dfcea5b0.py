import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra import numpy as npst

@given(npst.arrays(dtype=npst.integer_dtypes(), shape=npst.array_shapes()),
       st.data())
@settings(max_examples=100)
def test_unique_treats_masked_as_equal(arr, data):
    assume(arr.size > 1)
    mask = data.draw(npst.arrays(dtype=np.bool_, shape=arr.shape))
    assume(np.sum(mask) >= 2)

    marr = ma.array(arr, mask=mask)

    unique_result = ma.unique(marr)

    masked_in_result = ma.getmaskarray(unique_result)
    num_masked = np.sum(masked_in_result)

    # According to documentation: "Masked values are considered the same element (masked)"
    # This means all masked values should collapse to at most 1 masked value in the output
    assert num_masked <= 1, f"Expected at most 1 masked value, but got {num_masked}. Input: arr={arr}, mask={mask}"

if __name__ == "__main__":
    # Run the test
    test_unique_treats_masked_as_equal()