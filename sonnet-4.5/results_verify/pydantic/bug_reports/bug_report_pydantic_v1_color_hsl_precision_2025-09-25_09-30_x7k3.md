# Bug Report: pydantic.v1.Color HSL String Precision Loss

**Target**: `pydantic.v1.color.Color.as_hsl()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Color.as_hsl()` method loses precision for very dark or very light colors due to formatting HSL percentages with zero decimal places, breaking round-trip conversion.

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
```

**Failing input**: `r=0, g=0, b=2` (and many other very dark colors)

## Reproducing the Bug

```python
from pydantic.v1.color import Color

color1 = Color((0, 0, 2))
print(f"Original: {color1.as_rgb_tuple()}")

hsl_str = color1.as_hsl()
print(f"HSL: {hsl_str}")

color2 = Color(hsl_str)
print(f"After round-trip: {color2.as_rgb_tuple()}")
```

**Output:**
```
Original: (0, 0, 2)
HSL: hsl(240, 100%, 0%)
After round-trip: (0, 0, 0)
```

The blue component (value 2) is completely lost, becoming 0.

## Why This Is A Bug

The `as_hsl()` method formats saturation and lightness percentages with zero decimal places (`{s:0.0%}, {li:0.0%}`). For very dark colors like RGB(0, 0, 2), the lightness is approximately 0.39%, which rounds to 0% when formatted. When this "hsl(240, 100%, 0%)" string is parsed back, it becomes pure black RGB(0, 0, 0).

This violates the reasonable expectation that:
- `Color(color.as_hsl()) ≈ color` (round-trip property)
- Color information should not be silently lost in string representations

**Impact:** Users working with very dark or very light colors will experience data loss when using HSL string representation for serialization, display, or any round-trip operations.

## Fix

Increase the decimal precision in HSL percentage formatting to preserve color information. CSS3 allows decimal places in percentages, so this is standards-compliant.

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

This changes the format from e.g. "hsl(240, 100%, 0%)" to "hsl(240.0, 100.0%, 0.4%)", preserving precision while remaining CSS3-compliant.