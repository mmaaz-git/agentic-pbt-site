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
    # Try to find a failing example
    import sys
    found_failure = False
    test_cases = [
        {'0': ''},
        {'a': 1},
        {'b': 'text'},
        {'foo': 'bar'},
    ]

    for test_input in test_cases:
        try:
            pandas_dtype(test_input)
            print(f"Input {test_input} was accepted (no exception)")
        except TypeError as e:
            print(f"Input {test_input} raised TypeError (expected): {e}")
        except ValueError as e:
            print(f"BUG CONFIRMED: Input {test_input} raised ValueError instead of TypeError: {e}")
            found_failure = True

    sys.exit(0 if found_failure else 1)