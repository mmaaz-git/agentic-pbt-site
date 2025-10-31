import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from pydantic.color import Color

# Test the specific example
color = Color((0, 0, 22))
print(f"Original RGB: {color.as_rgb_tuple()}")

hsl_str = color.as_hsl()
print(f"HSL string: {hsl_str}")

color2 = Color(hsl_str)
print(f"After round-trip: {color2.as_rgb_tuple()}")

print(f"Difference: {abs(22 - color2.as_rgb_tuple()[2])}")

print("\nAdditional examples:")
# Test (0, 0, 1)
color = Color((0, 0, 1))
hsl_str = color.as_hsl()
color2 = Color(hsl_str)
print(f"(0, 0, 1) → {hsl_str} → {color2.as_rgb_tuple()} (diff: {abs(1 - color2.as_rgb_tuple()[2])})")

# Test (0, 0, 2)
color = Color((0, 0, 2))
hsl_str = color.as_hsl()
color2 = Color(hsl_str)
print(f"(0, 0, 2) → {hsl_str} → {color2.as_rgb_tuple()} (diff: {abs(2 - color2.as_rgb_tuple()[2])})")

# Test (0, 0, 22)
color = Color((0, 0, 22))
hsl_str = color.as_hsl()
color2 = Color(hsl_str)
print(f"(0, 0, 22) → {hsl_str} → {color2.as_rgb_tuple()} (diff: {abs(22 - color2.as_rgb_tuple()[2])})")

# Test more examples to understand the pattern
print("\nMore examples to understand precision loss:")
test_cases = [
    (0, 0, 3),
    (0, 0, 5),
    (0, 0, 10),
    (0, 0, 15),
    (0, 0, 20),
    (0, 0, 25),
    (0, 0, 50),
    (0, 0, 100),
    (0, 0, 128),
    (0, 0, 200),
    (0, 0, 255),
]

for rgb in test_cases:
    color = Color(rgb)
    hsl_str = color.as_hsl()
    color2 = Color(hsl_str)
    rgb2 = color2.as_rgb_tuple()
    diff = abs(rgb[2] - rgb2[2])
    if diff > 0:
        print(f"{rgb} → {hsl_str} → {rgb2} (diff: {diff})")

# Calculate what the actual HSL values are for (0, 0, 22)
print("\nDetailed calculation for (0, 0, 22):")
from colorsys import rgb_to_hls
r, g, b = 0/255, 0/255, 22/255
h, l, s = rgb_to_hls(r, g, b)
print(f"Exact HSL values: h={h}, s={s}, l={l}")
print(f"In degrees/percentage: h={h*360}°, s={s*100}%, l={l*100}%")
print(f"Formatted with {'{:0.0%}'}: s={s:0.0%}, l={l:0.0%}")
print(f"Formatted with {'{:0.1%}'}: s={s:0.1%}, l={l:0.1%}")