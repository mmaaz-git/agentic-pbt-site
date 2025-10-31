import numpy as np
import numpy.ma as ma
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra import numpy as nps


@st.composite
def identical_masked_arrays_with_some_masked(draw):
    size = draw(st.integers(min_value=2, max_value=30))
    data = draw(nps.arrays(dtype=np.float64, shape=(size,),
                          elements={"allow_nan": False, "allow_infinity": False,
                                   "min_value": -100, "max_value": 100}))
    mask = draw(nps.arrays(dtype=bool, shape=(size,)))
    assume(mask.any())
    assume((~mask).any())
    return data, mask


@given(identical_masked_arrays_with_some_masked())
@settings(max_examples=500)
def test_allequal_fillvalue_false_bug(data_mask):
    data, mask = data_mask
    x = ma.array(data, mask=mask)
    y = ma.array(data.copy(), mask=mask.copy())

    result_false = ma.allequal(x, y, fill_value=False)
    unmasked_equal = np.array_equal(data[~mask], data[~mask])

    if unmasked_equal:
        assert result_false == True, \
            f"allequal with fill_value=False returned False for arrays with identical unmasked values\nData: {data}\nMask: {mask}"

if __name__ == "__main__":
    # Run the test
    test_allequal_fillvalue_false_bug()
    print("Test completed")