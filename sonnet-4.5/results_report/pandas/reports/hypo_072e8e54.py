import pandas.api.types as types
from hypothesis import given, strategies as st


@given(st.one_of(
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans(),
))
def test_infer_dtype_accepts_scalars(val):
    result_scalar = types.infer_dtype(val, skipna=False)
    result_list = types.infer_dtype([val], skipna=False)
    assert result_scalar == result_list

if __name__ == "__main__":
    test_infer_dtype_accepts_scalars()