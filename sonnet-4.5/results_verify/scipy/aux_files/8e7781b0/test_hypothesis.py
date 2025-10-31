from hypothesis import given, strategies as st, assume
import numpy as np
from xarray.core.indexes import normalize_label

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100),
                min_size=1, max_size=20, unique=True))
def test_normalize_label_with_float32_dtype(float_values):
    assume(len(float_values) > 0)
    float32_arr = np.array(float_values, dtype=np.float32)
    result = normalize_label(float32_arr, dtype=np.float32)
    assert result.dtype == np.float32

if __name__ == "__main__":
    test_normalize_label_with_float32_dtype()