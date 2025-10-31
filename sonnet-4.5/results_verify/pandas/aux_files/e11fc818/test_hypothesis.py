from hypothesis import given, strategies as st, settings
import pandas.api.types as pt

@given(st.text(min_size=1, max_size=10))
@settings(max_examples=1000)
def test_pandas_dtype_raises_typeerror_on_invalid_input(s):
    try:
        pt.pandas_dtype(s)
    except TypeError:
        pass
    except ValueError as e:
        raise AssertionError(f"pandas_dtype raised ValueError instead of documented TypeError for input {s!r}: {e}")

# Run the test
try:
    test_pandas_dtype_raises_typeerror_on_invalid_input()
    print("Test passed!")
except AssertionError as e:
    print(f"Test failed: {e}")