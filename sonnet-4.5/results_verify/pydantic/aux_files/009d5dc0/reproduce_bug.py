from pydantic.v1.color import Color

color1 = Color((0, 0, 2))
print(f"Original: {color1.as_rgb_tuple()}")

hsl_str = color1.as_hsl()
print(f"HSL: {hsl_str}")

color2 = Color(hsl_str)
print(f"After round-trip: {color2.as_rgb_tuple()}")