from hypothesis import given, strategies as st
import pytest
from django.template import Variable

@given(st.floats(allow_nan=False, allow_infinity=False, min_value=1, max_value=1e10))
def test_variable_float_with_trailing_dot_should_be_rejected(num):
    """
    Property: Floats with trailing dots should be rejected as invalid.
    Evidence: Code comment on line 824 says '"2." is invalid' and code
    explicitly raises ValueError for this case on line 826.
    """
    var_str = f"{int(num)}."

    with pytest.raises((ValueError, Exception)):
        var = Variable(var_str)

# Run the test
if __name__ == "__main__":
    test_variable_float_with_trailing_dot_should_be_rejected()