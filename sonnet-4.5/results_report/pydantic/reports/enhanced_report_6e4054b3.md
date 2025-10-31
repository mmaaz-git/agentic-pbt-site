# Bug Report: pydantic.v1.Color HSL String Precision Loss

**Target**: `pydantic.v1.color.Color.as_hsl()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Color.as_hsl()` method loses precision for very dark or very light colors due to formatting HSL percentages with zero decimal places, causing round-trip conversion failures and silent data loss.

## Property-Based Test

```python
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

# Run the test
if __name__ == "__main__":
    test_color_hsl_roundtrip()
```

<details>

<summary>
**Failing input**: `r=0, g=0, b=104`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 21, in <module>
    test_color_hsl_roundtrip()
    ~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 7, in test_color_hsl_roundtrip
    def test_color_hsl_roundtrip(r, g, b):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 16, in test_color_hsl_roundtrip
    assert abs(rgb1[i] - rgb2[i]) <= 1, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: HSL round-trip failed: (0, 0, 104) != (0, 0, 102)
Falsifying example: test_color_hsl_roundtrip(
    r=0,
    g=0,
    b=104,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/18/hypo.py:17
```
</details>

## Reproducing the Bug

```python
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
```

<details>

<summary>
Round-trip conversion fails, losing blue component value
</summary>
```
Original RGB: (0, 0, 2)
HSL string: hsl(240, 100%, 0%)
After round-trip: (0, 0, 0)
✗ Round-trip FAILED: (0, 0, 2) != (0, 0, 0)
  Lost color components!
```
</details>

## Why This Is A Bug

This bug violates the reasonable expectation that color information should be preserved when converting between equivalent representations. The issue occurs because:

1. **Precision Loss in Format String**: The `as_hsl()` method uses format specifiers `{s:0.0%}` and `{li:0.0%}` (lines 159 and 162 in `/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages/pydantic/v1/color.py`), which format saturation and lightness percentages with zero decimal places.

2. **Silent Data Loss**: For RGB(0, 0, 2), the lightness value is approximately 0.39% (calculated as 2/255 ≈ 0.0078, which converts to ~0.39% in HSL). When formatted with zero decimal places, this rounds down to 0%, completely losing the color information.

3. **Asymmetric Behavior**: The Color class can correctly parse HSL strings with decimal percentages (the regex patterns on lines 54-56 accept decimal values), but it outputs integers only. This asymmetry suggests an oversight rather than intentional design.

4. **CSS3 Compliance**: The CSS3 specification explicitly allows decimal places in HSL percentage values, so using higher precision would remain standards-compliant while preserving color data.

5. **Inconsistent with Other Methods**: Other color representations (`as_rgb()`, `as_hex()`) preserve full precision, making HSL an outlier in losing data unnecessarily.

## Relevant Context

The bug affects colors with RGB components in the ranges:
- Very dark colors: RGB values 0-2 (lightness < 0.5%)
- Very light colors: RGB values 253-255 (lightness > 99.5%)
- Low saturation colors near grayscale

This is particularly problematic for:
- Scientific visualization requiring precise color gradients
- Image processing where subtle color differences matter
- Color theme systems with dark/light variations
- Any application requiring reliable color serialization

The CSS3 Color Module specification (referenced in the code at line 3) does not restrict HSL percentages to integers. Many CSS implementations and browsers handle decimal percentages correctly.

## Proposed Fix

Increase decimal precision in HSL percentage formatting to preserve color information while maintaining CSS3 compliance:

```diff
--- a/pydantic/v1/color.py
+++ b/pydantic/v1/color.py
@@ -156,11 +156,11 @@ class Color(Representation):
         """
         if self._rgba.alpha is None:
             h, s, li = self.as_hsl_tuple(alpha=False)  # type: ignore
-            return f'hsl({h * 360:0.0f}, {s:0.0%}, {li:0.0%})'
+            return f'hsl({h * 360:0.1f}, {s:0.1%}, {li:0.1%})'
         else:
             h, s, li, a = self.as_hsl_tuple(alpha=True)  # type: ignore
-            return f'hsl({h * 360:0.0f}, {s:0.0%}, {li:0.0%}, {round(a, 2)})'
+            return f'hsl({h * 360:0.1f}, {s:0.1%}, {li:0.1%}, {round(a, 2)})'

     def as_hsl_tuple(self, *, alpha: Optional[bool] = None) -> HslColorTuple:
```