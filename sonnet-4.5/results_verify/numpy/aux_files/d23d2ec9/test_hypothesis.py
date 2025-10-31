import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st, settings

@given(
    st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=20)
)
@settings(max_examples=500)
def test_shrink_with_scalar_false_mask(data_list):
    data = np.array(data_list)

    arr_with_npfalse = ma.array(data, mask=np.False_, shrink=True)
    arr_with_pyfalse = ma.array(data, mask=False, shrink=True)

    mask_npfalse = ma.getmask(arr_with_npfalse)
    mask_pyfalse = ma.getmask(arr_with_pyfalse)

    assert mask_npfalse is ma.nomask
    assert mask_pyfalse is ma.nomask

if __name__ == "__main__":
    # Run the test
    test_shrink_with_scalar_false_mask()