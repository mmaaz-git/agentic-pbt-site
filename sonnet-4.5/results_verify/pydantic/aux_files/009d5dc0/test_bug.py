from hypothesis import given, strategies as st
from pydantic.v1.color import Color

rgb_components = st.integers(min_value=0, max_value=255)

@given(rgb_components, rgb_components, rgb_components)
def test_color_hsl_roundtrip(r, g, b):
    color1 = Color((r, g, b))
    hsl_str = color1.as_hsl()
    color2 = Color(hsl_str)

    rgb1 = color1.as_rgb_tuple(alpha=False)
    rgb2 = color2.as_rgb_tuple(alpha=False)

    for i in range(3):
        assert abs(rgb1[i] - rgb2[i]) <= 1, \
            f"HSL round-trip failed: {rgb1} != {rgb2}"

if __name__ == "__main__":
    # Run the test
    test_color_hsl_roundtrip()