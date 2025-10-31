from hypothesis import given, strategies as st, settings, example
from pandas.io.formats.css import CSSResolver
import math


@settings(max_examples=1000)
@given(st.floats(min_value=0.000001, max_value=1e6, allow_nan=False, allow_infinity=False))
@example(6.103515625e-05)  # The specific failing case mentioned
def test_pt_to_pt_should_preserve_value(value):
    resolver = CSSResolver()
    input_str = f"{value}pt"
    result = resolver.size_to_pt(input_str)
    result_val = float(result.rstrip("pt"))

    # Debug output for failing cases
    if not (math.isclose(result_val, value, abs_tol=1e-5) or result_val == value):
        print(f"Failed for value: {value}")
        print(f"Input string: {input_str}")
        print(f"Result: {result}")
        print(f"Result value: {result_val}")
        print(f"Expected: {value}")

    assert math.isclose(result_val, value, abs_tol=1e-5) or result_val == value

# Run the test
if __name__ == "__main__":
    test_pt_to_pt_should_preserve_value()