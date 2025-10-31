from hypothesis import given, strategies as st, settings
from pandas.core.dtypes.common import ensure_python_int
import pytest
import numpy as np


@given(st.floats(allow_nan=True, allow_infinity=True))
@settings(max_examples=200)
def test_ensure_python_int_special_floats(value):
    if np.isnan(value) or np.isinf(value):
        with pytest.raises(TypeError):
            ensure_python_int(value)
    elif value == int(value):
        result = ensure_python_int(value)
        assert result == int(value)


if __name__ == "__main__":
    test_ensure_python_int_special_floats()