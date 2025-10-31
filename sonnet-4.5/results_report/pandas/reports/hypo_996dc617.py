from pandas.core.dtypes.common import ensure_python_int
from hypothesis import given, strategies as st

@given(st.floats(allow_infinity=True))
def test_ensure_python_int_exception_contract(value):
    try:
        result = ensure_python_int(value)
        assert isinstance(result, int)
        assert result == value
    except TypeError:
        pass
    except OverflowError:
        raise AssertionError(
            f"ensure_python_int raised OverflowError instead of TypeError for {value}"
        )

if __name__ == "__main__":
    test_ensure_python_int_exception_contract()