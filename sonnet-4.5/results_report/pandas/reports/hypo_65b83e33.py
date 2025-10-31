from pandas.core.dtypes.common import ensure_python_int
from hypothesis import given, strategies as st, settings
import pytest


@given(st.one_of(st.just(float('inf')), st.just(float('-inf')), st.just(float('nan'))))
@settings(max_examples=10)
def test_ensure_python_int_special_floats_raise_typeerror(x):
    with pytest.raises(TypeError):
        ensure_python_int(x)


if __name__ == "__main__":
    # Run the test
    test_ensure_python_int_special_floats_raise_typeerror()