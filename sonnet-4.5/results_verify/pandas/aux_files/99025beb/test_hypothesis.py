from hypothesis import given, strategies as st
import pytest
from pandas.core.computation.eval import _check_expression

@given(st.just("") | st.text().filter(lambda x: x.isspace()))
def test_check_expression_rejects_empty_string(expr):
    with pytest.raises(ValueError, match="expr cannot be an empty string"):
        _check_expression(expr)

# Run the test
if __name__ == "__main__":
    test_check_expression_rejects_empty_string()