from hypothesis import given, strategies as st
import numpy as np
from pandas.core.dtypes.dtypes import SparseDtype

@st.composite
def valid_sparse_dtypes(draw):
    base_dtype = draw(st.sampled_from([np.float32, np.float64]))
    use_default = draw(st.booleans())
    if use_default:
        return SparseDtype(base_dtype)
    fill_value = draw(st.one_of(
        st.floats(allow_nan=True, allow_infinity=True),
        st.floats(allow_nan=False, allow_infinity=False)
    ))
    return SparseDtype(base_dtype, fill_value)

@given(valid_sparse_dtypes(), valid_sparse_dtypes())
def test_equality_symmetric(dtype1, dtype2):
    """Property: If dtype1 == dtype2, then dtype2 == dtype1"""
    if dtype1 == dtype2:
        assert dtype2 == dtype1, \
            f"Equality not symmetric: {dtype1} == {dtype2} but {dtype2} != {dtype1}"

if __name__ == "__main__":
    test_equality_symmetric()