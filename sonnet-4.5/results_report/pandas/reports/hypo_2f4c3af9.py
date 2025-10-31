from hypothesis import given, strategies as st
import pandas as pd
import numpy as np

@given(st.lists(st.floats(min_value=-1.0, max_value=-0.01), min_size=1, max_size=10))
def test_negative_weights_error_message(negative_weights):
    """Error message should say 'may not' not 'many not'"""
    df = pd.DataFrame({'A': range(len(negative_weights))})
    weights = pd.Series(negative_weights)

    try:
        df.sample(n=1, weights=weights)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        error_msg = str(e)
        assert "may not" in error_msg or "many not" in error_msg
        if "many not" in error_msg:
            raise AssertionError(f"Typo in error message: {error_msg}")

# Run the test
test_negative_weights_error_message()