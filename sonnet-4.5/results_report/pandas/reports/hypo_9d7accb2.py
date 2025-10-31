from hypothesis import given, strategies as st, settings, HealthCheck, example
import pandas as pd
import pytest

# Strategy for generating whitespace-only strings
whitespace_strategy = st.text(alphabet=' \t\n\r', min_size=1, max_size=20)

@given(whitespace_strategy)
@settings(suppress_health_check=[HealthCheck.filter_too_much])
@example("   ")  # Basic spaces
@example("\t\t")  # Tabs
@example("\n\n")  # Newlines
@example("   \n\t  ")  # Mixed whitespace
def test_eval_whitespace_only_expression_should_raise(whitespace_expr):
    """Test that pd.eval raises ValueError for whitespace-only expressions"""
    # Only test if the string is truly whitespace-only
    if whitespace_expr.strip() == "":
        with pytest.raises(ValueError, match="expr cannot be an empty string"):
            pd.eval(whitespace_expr)

# Run the test
if __name__ == "__main__":
    # Run the test and capture the failure
    test_eval_whitespace_only_expression_should_raise()