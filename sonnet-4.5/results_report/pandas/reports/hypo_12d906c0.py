from hypothesis import given, strategies as st, example
from pandas.core.dtypes.common import ensure_python_int
import pytest


@given(st.floats(allow_nan=True, allow_infinity=True))
@example(float('inf'))
@example(float('-inf'))
def test_ensure_python_int_raises_typeerror_for_invalid_floats(value):
    try:
        int_value = int(value)
        if value != int_value:
            with pytest.raises(TypeError):
                ensure_python_int(value)
    except (OverflowError, ValueError):
        # For inf and nan, we expect TypeError from ensure_python_int
        with pytest.raises(TypeError):
            ensure_python_int(value)