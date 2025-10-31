from hypothesis import given, strategies as st
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

# Run the hypothesis test
@given(
    r=st.integers(min_value=0, max_value=255),
    g=st.integers(min_value=0, max_value=255),
    b=st.integers(min_value=0, max_value=255)
)
def test_color_hsl_roundtrip(r, g, b):
    color = Color((r, g, b))
    hsl_str = color.as_hsl()
    color2 = Color(hsl_str)

    t1 = color.as_rgb_tuple()
    t2 = color2.as_rgb_tuple()

    # Check if values differ by more than 2
    if not all(abs(a - b) <= 2 for a, b in zip(t1, t2)):
        print(f"\nLarge difference found: {t1} -> {hsl_str} -> {t2}")
        return False
    return True

# Run a few tests
print("\nRunning hypothesis tests...")
import random
random.seed(42)
for _ in range(100):
    r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
    test_color_hsl_roundtrip(r, g, b)