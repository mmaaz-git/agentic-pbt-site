from hypothesis import given, strategies as st
from pandas.io.formats.css import CSSResolver

resolver = CSSResolver()

@given(
    val=st.floats(min_value=1e-10, max_value=1e10, allow_nan=False, allow_infinity=False),
    unit=st.sampled_from(["pt", "px", "em", "rem", "in", "cm", "mm"])
)
def test_size_to_pt_scientific_notation(val, unit):
    input_str = f"{val}{unit}"
    result = resolver.size_to_pt(input_str)
    assert result.endswith("pt"), f"Result {result} should end with 'pt'"
    result_val = float(result.rstrip("pt"))
    assert result_val != 0 or val == 0, f"Non-zero input {input_str} should not produce 0pt"

# Run the test
if __name__ == "__main__":
    test_size_to_pt_scientific_notation()