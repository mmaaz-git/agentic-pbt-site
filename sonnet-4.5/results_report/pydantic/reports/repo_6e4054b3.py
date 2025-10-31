from pydantic.v1.color import Color

# Test case with very dark blue color
color1 = Color((0, 0, 2))
print(f"Original RGB: {color1.as_rgb_tuple()}")

# Convert to HSL string representation
hsl_str = color1.as_hsl()
print(f"HSL string: {hsl_str}")

# Parse the HSL string back to create a new color
color2 = Color(hsl_str)
print(f"After round-trip: {color2.as_rgb_tuple()}")

# Check if colors match
rgb1 = color1.as_rgb_tuple(alpha=False)
rgb2 = color2.as_rgb_tuple(alpha=False)

if rgb1 == rgb2:
    print(f"✓ Round-trip successful: {rgb1} == {rgb2}")
else:
    print(f"✗ Round-trip FAILED: {rgb1} != {rgb2}")
    print(f"  Lost color components!")