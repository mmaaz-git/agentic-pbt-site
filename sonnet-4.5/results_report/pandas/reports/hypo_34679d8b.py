import pandas as pd
from hypothesis import given, strategies as st, settings


@settings(max_examples=100)
@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100), min_size=5, max_size=20),
    st.integers(min_value=0, max_value=5)
)
def test_rolling_step_validation(data, step):
    """
    Property: Creating a rolling window with any step value should either
    succeed and allow aggregations, or fail during validation with a clear error.
    It should NOT pass validation and then crash during computation.
    """
    df = pd.DataFrame({'A': data})

    try:
        rolling = df.rolling(window=2, step=step)
        result = rolling.mean()
        assert result is not None
    except ValueError as e:
        # This should be raised during validation, not computation
        if "step must be" in str(e):
            # This is expected validation error - good!
            pass
        else:
            # If it's some other ValueError during computation, that's bad
            raise AssertionError(
                f"step={step} passed validation but crashed during computation with: {e}"
            )
    except ZeroDivisionError as e:
        # ZeroDivisionError should never happen - it means validation failed
        raise AssertionError(
            f"step={step} passed validation but crashed during computation with ZeroDivisionError: {e}"
        )

if __name__ == "__main__":
    # Run the test
    test_rolling_step_validation()