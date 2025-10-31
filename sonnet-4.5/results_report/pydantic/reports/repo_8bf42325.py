from pydantic.color import Color

rgba = (0, 0, 0, 0.625)
color = Color(rgba)
rgba_str = color.as_rgb()
color2 = Color(rgba_str)

print(f"Original: {color.as_rgb_tuple(alpha=True)}")
print(f"RGBA string: {rgba_str}")
print(f"After round-trip: {color2.as_rgb_tuple(alpha=True)}")
print(f"Match: {color == color2}")