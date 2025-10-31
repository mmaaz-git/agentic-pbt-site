from pandas.api.types import pandas_dtype
from hypothesis import given, strategies as st
import pytest


@given(st.dictionaries(
    st.text(min_size=1, max_size=10),
    st.one_of(st.integers(), st.text()),
    min_size=1,
    max_size=10
))
def test_pandas_dtype_raises_only_typeerror(d):
    """pandas_dtype should only raise TypeError for invalid inputs, per docstring."""
    try:
        pandas_dtype(d)
    except TypeError:
        pass
    except ValueError as e:
        pytest.fail(
            f"pandas_dtype raised ValueError instead of TypeError for input {d!r}.\n"
            f"Docstring says it should only raise TypeError.\n"
            f"Error was: {e}"
        )

if __name__ == "__main__":
    test_pandas_dtype_raises_only_typeerror()