import pytest
from hypothesis import given, settings, strategies as st
from pandas.io.parsers.readers import validate_integer


@settings(max_examples=500)
@given(
    val=st.floats(allow_nan=False, allow_infinity=False),
    min_val=st.integers(min_value=0, max_value=1000)
)
def test_validate_integer_respects_min_val_for_floats(val, min_val):
    if val != int(val):
        return

    if int(val) >= min_val:
        result = validate_integer("test", val, min_val)
        assert result >= min_val
    else:
        with pytest.raises(ValueError, match="must be an integer"):
            validate_integer("test", val, min_val)

if __name__ == "__main__":
    # Run the test
    test_validate_integer_respects_min_val_for_floats()