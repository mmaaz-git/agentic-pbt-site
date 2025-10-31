from pydantic.color import Color

rgb = (0, 0, 1)
color = Color(rgb)
hsl_str = color.as_hsl()
color2 = Color(hsl_str)

print(f"Original: {color.as_rgb_tuple()}")
print(f"HSL: {hsl_str}")
print(f"After round-trip: {color2.as_rgb_tuple()}")
print(f"Match: {color == color2}")

# Let's check the actual HSL values before formatting
h, s, l = color.as_hsl_tuple(alpha=False)
print(f"\nActual HSL values (0-1 range):")
print(f"  Hue: {h}")
print(f"  Saturation: {s}")
print(f"  Lightness: {l}")
print(f"\nFormatted in HSL string:")
print(f"  Hue: {h * 360:0.0f} degrees")
print(f"  Saturation: {s:0.0%}")
print(f"  Lightness: {l:0.0%}")

# Calculate what the actual lightness percentage is
actual_lightness_percent = l * 100
print(f"\nActual lightness as percentage: {actual_lightness_percent}%")
print(f"Rounded to 0 decimal places: {actual_lightness_percent:0.0f}%")