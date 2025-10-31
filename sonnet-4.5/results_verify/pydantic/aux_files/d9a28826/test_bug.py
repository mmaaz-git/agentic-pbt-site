from hypothesis import given, strategies as st, settings
from pydantic.color import Color

rgb_values = st.integers(min_value=0, max_value=255)

@given(st.tuples(rgb_values, rgb_values, rgb_values))
@settings(max_examples=1000)
def test_hsl_round_trip(rgb):
    color = Color(rgb)
    hsl_str = color.as_hsl()
    color2 = Color(hsl_str)
    assert color == color2

# Run the test
if __name__ == "__main__":
    test_hsl_round_trip()