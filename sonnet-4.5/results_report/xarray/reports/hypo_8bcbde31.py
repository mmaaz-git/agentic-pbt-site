import numpy as np
from hypothesis import given, strategies as st, settings
from xarray.core import duck_array_ops


@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1, max_size=100),
    st.lists(st.booleans(), min_size=1, max_size=100)
)
@settings(max_examples=100)
def test_sum_where_matches_numpy(data_list, where_list):
    size = min(len(data_list), len(where_list))
    data = np.array(data_list[:size])
    where = np.array(where_list[:size])

    numpy_result = np.sum(data, where=where)
    xarray_result = duck_array_ops.sum_where(data, where=where)

    assert np.isclose(numpy_result, xarray_result), f"numpy: {numpy_result}, xarray: {xarray_result}, data: {data_list[:size]}, where: {where_list[:size]}"

if __name__ == "__main__":
    test_sum_where_matches_numpy()