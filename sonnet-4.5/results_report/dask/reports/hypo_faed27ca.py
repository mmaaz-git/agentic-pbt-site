import numpy as np
import dask.array as da
from hypothesis import given, strategies as st, settings
from hypothesis.extra import numpy as hnp


@st.composite
def dask_array_for_argtopk(draw):
    shape = draw(st.tuples(st.integers(min_value=5, max_value=30)))
    dtype = draw(st.sampled_from([np.int32, np.float64]))

    if dtype == np.float64:
        np_arr = draw(hnp.arrays(
            dtype, shape,
            elements=st.floats(min_value=-1000, max_value=1000,
                             allow_nan=False, allow_infinity=False)
        ))
    else:
        np_arr = draw(hnp.arrays(
            dtype, shape,
            elements=st.integers(min_value=-1000, max_value=1000)
        ))

    chunks = draw(st.integers(min_value=2, max_value=max(3, shape[0] // 2)))
    k = draw(st.integers(min_value=1, max_value=min(10, shape[0])))

    return da.from_array(np_arr, chunks=chunks), k


@given(dask_array_for_argtopk())
@settings(max_examples=200)
def test_argtopk_returns_correct_size(data):
    arr, k = data
    result = da.argtopk(arr, k).compute()
    assert len(result) == k


if __name__ == "__main__":
    test_argtopk_returns_correct_size()