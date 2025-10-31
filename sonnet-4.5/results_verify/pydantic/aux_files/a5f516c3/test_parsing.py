import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from pydantic.color import Color
from colorsys import hls_to_rgb

# What does 4% lightness convert back to?
print("Testing HSL parsing:")
print("When we parse 'hsl(240, 100%, 4%)':")

# Parse the HSL string
color = Color("hsl(240, 100%, 4%)")
rgb = color.as_rgb_tuple()
print(f"Result: RGB {rgb}")

# Calculate what 4% lightness should give us
h_norm = 240/360  # normalized hue
s_norm = 1.0      # 100% saturation
l_norm = 0.04     # 4% lightness

r, g, b = hls_to_rgb(h_norm, l_norm, s_norm)
print(f"\nDirect calculation with 4% lightness:")
print(f"Float RGB: ({r}, {g}, {b})")
print(f"Integer RGB: ({int(round(r*255))}, {int(round(g*255))}, {int(round(b*255))})")

# What about 4.3% (the actual value)?
l_norm = 0.043137254901960784  # The actual lightness for RGB(0,0,22)
r, g, b = hls_to_rgb(h_norm, l_norm, s_norm)
print(f"\nDirect calculation with 4.3137% lightness:")
print(f"Float RGB: ({r}, {g}, {b})")
print(f"Integer RGB: ({int(round(r*255))}, {int(round(g*255))}, {int(round(b*255))})")

# Test the round-trip for several values
print("\n\nRound-trip precision loss analysis:")
print("="*50)

def analyze_roundtrip(rgb_tuple):
    from colorsys import rgb_to_hls

    color1 = Color(rgb_tuple)

    # Get exact HSL values
    r, g, b = [v/255 for v in rgb_tuple]
    h, l, s = rgb_to_hls(r, g, b)

    # Get the string representation
    hsl_str = color1.as_hsl()

    # Parse it back
    color2 = Color(hsl_str)
    rgb2 = color2.as_rgb_tuple()

    print(f"\nOriginal RGB: {rgb_tuple}")
    print(f"Exact L value: {l:.10f} ({l*100:.6f}%)")
    print(f"HSL string: {hsl_str}")
    print(f"Parsed back: {rgb2}")
    print(f"Difference: {tuple(abs(a-b) for a,b in zip(rgb_tuple, rgb2))}")

    return rgb2

# Test a few values
analyze_roundtrip((0, 0, 20))
analyze_roundtrip((0, 0, 21))
analyze_roundtrip((0, 0, 22))
analyze_roundtrip((0, 0, 23))