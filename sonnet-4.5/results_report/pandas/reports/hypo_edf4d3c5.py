from hypothesis import given, strategies as st, settings
from pandas.io.parsers.readers import validate_integer

@settings(max_examples=1000)
@given(st.integers(max_value=-1), st.integers(min_value=0, max_value=100))
def test_validate_integer_min_val_consistency_int_vs_float(val, min_val):
    int_raised = False
    float_raised = False

    try:
        int_result = validate_integer("test", val, min_val=min_val)
    except ValueError:
        int_raised = True

    try:
        float_result = validate_integer("test", float(val), min_val=min_val)
    except ValueError:
        float_raised = True

    if val < min_val:
        assert int_raised, f"Integer {val} should raise ValueError"
        assert float_raised, f"Float {float(val)} should raise ValueError"
        assert int_raised == float_raised, "Inconsistent behavior"

# Run the test
if __name__ == "__main__":
    test_validate_integer_min_val_consistency_int_vs_float()