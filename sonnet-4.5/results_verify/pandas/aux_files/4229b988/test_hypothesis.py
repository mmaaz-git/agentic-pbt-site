from pandas.core.dtypes.common import ensure_python_int
from hypothesis import given, strategies as st
import traceback

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

# Run the test
try:
    test_ensure_python_int_exception_contract()
    print("Hypothesis test passed")
except AssertionError as e:
    print(f"Hypothesis test failed with AssertionError: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"Hypothesis test failed with unexpected error: {e}")
    traceback.print_exc()