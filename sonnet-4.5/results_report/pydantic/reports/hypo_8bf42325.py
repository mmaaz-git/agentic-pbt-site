from hypothesis import given, strategies as st, settings
from pydantic.color import Color

rgb_values = st.integers(min_value=0, max_value=255)
alpha_values = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

@given(st.tuples(rgb_values, rgb_values, rgb_values, alpha_values))
@settings(max_examples=1000)
def test_rgba_string_round_trip(rgba):
    color = Color(rgba)
    rgba_str = color.as_rgb()
    color2 = Color(rgba_str)
    assert color == color2, f"Round-trip failed for {rgba}: {color.as_rgb_tuple(alpha=True)} != {color2.as_rgb_tuple(alpha=True)}"

if __name__ == "__main__":
    test_rgba_string_round_trip()