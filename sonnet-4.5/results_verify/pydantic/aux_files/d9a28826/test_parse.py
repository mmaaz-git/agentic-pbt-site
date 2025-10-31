from pydantic.color import Color

# Parse the HSL string
color = Color("hsl(240, 100%, 0%)")
print(f"Parsing 'hsl(240, 100%, 0%)' gives RGB: {color.as_rgb_tuple()}")

# What about a very small but non-zero lightness?
color2 = Color("hsl(240, 100%, 0.2%)")
print(f"Parsing 'hsl(240, 100%, 0.2%)' gives RGB: {color2.as_rgb_tuple()}")

# And 0.1%?
color3 = Color("hsl(240, 100%, 0.1%)")
print(f"Parsing 'hsl(240, 100%, 0.1%)' gives RGB: {color3.as_rgb_tuple()}")

# Let's test some other edge cases
test_cases = [
    (0, 1, 0),  # Very dark green
    (1, 0, 0),  # Very dark red
    (1, 1, 0),  # Very dark yellow
    (0, 0, 2),  # Slightly brighter blue
    (2, 0, 0),  # Slightly brighter red
]

print("\nOther edge cases:")
for rgb in test_cases:
    c1 = Color(rgb)
    hsl = c1.as_hsl()
    c2 = Color(hsl)
    match = c1 == c2
    print(f"RGB {rgb} -> HSL '{hsl}' -> RGB {c2.as_rgb_tuple()}, Match: {match}")