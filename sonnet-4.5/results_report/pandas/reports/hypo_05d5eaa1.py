import numpy as np
from hypothesis import given, strategies as st
from pandas.core.internals.base import ensure_np_dtype


@given(st.sampled_from([np.dtype(str), np.dtype('U'), np.dtype('U10')]))
def test_ensure_np_dtype_string_to_object(str_dtype):
    result = ensure_np_dtype(str_dtype)
    assert isinstance(result, np.dtype), f"Expected np.dtype, got {type(result)}"
    assert result == np.dtype('object'), f"Expected object dtype, got {result}"


if __name__ == "__main__":
    test_ensure_np_dtype_string_to_object()