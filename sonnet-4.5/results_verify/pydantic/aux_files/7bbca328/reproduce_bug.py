from pydantic.color import Color

rgba = (0, 0, 0, 0.625)
color = Color(rgba)
rgba_str = color.as_rgb()
color2 = Color(rgba_str)

print(f"Original: {color.as_rgb_tuple(alpha=True)}")
print(f"RGBA string: {rgba_str}")
print(f"After round-trip: {color2.as_rgb_tuple(alpha=True)}")
print(f"Match: {color == color2}")

# Let's also test with the failing case from hypothesis
rgba2 = (0, 0, 0, 0.375)
color3 = Color(rgba2)
rgba_str2 = color3.as_rgb()
color4 = Color(rgba_str2)

print(f"\nOriginal 2: {color3.as_rgb_tuple(alpha=True)}")
print(f"RGBA string 2: {rgba_str2}")
print(f"After round-trip 2: {color4.as_rgb_tuple(alpha=True)}")
print(f"Match 2: {color3 == color4}")