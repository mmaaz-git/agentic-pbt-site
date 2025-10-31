from hypothesis import given, strategies as st
import pytest
from pandas.core.computation.eval import _check_expression

@given(st.just("") | st.text().filter(lambda x: x.isspace()))
def test_check_expression_rejects_empty_string(expr):
    """Test that _check_expression raises ValueError for empty and whitespace-only strings"""
    with pytest.raises(ValueError, match="expr cannot be an empty string"):
        _check_expression(expr)

# Run the test
if __name__ == "__main__":
    import sys
    try:
        test_check_expression_rejects_empty_string()
        print("Test passed - no issues found")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
