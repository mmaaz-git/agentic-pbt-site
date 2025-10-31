from hypothesis import given, strategies as st, settings
from pandas.core.dtypes.common import ensure_python_int

@given(st.one_of(st.just(float('inf')), st.just(float('-inf'))))
@settings(max_examples=20)
def test_ensure_python_int_infinity_raises_typeerror(value):
    try:
        result = ensure_python_int(value)
        assert False, f"Should have raised TypeError, got {result}"
    except TypeError:
        pass
    except Exception as e:
        assert False, f"Expected TypeError but got {type(e).__name__}: {e}"

if __name__ == "__main__":
    test_ensure_python_int_infinity_raises_typeerror()