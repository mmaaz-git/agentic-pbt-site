# Bug Report: pydantic.color.Color HSL Round-trip Data Loss

**Target**: `pydantic.color.Color.as_hsl()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Color.as_hsl()` method loses color information for very dark colors due to rounding saturation and lightness to 0 decimal places, violating the round-trip property `Color(color.as_hsl()) == color`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pydantic.color import Color

rgb_values = st.integers(min_value=0, max_value=255)

@given(st.tuples(rgb_values, rgb_values, rgb_values))
@settings(max_examples=1000)
def test_hsl_round_trip(rgb):
    color = Color(rgb)
    hsl_str = color.as_hsl()
    color2 = Color(hsl_str)
    assert color == color2
```

**Failing input**: `rgb=(0, 0, 1)`

## Reproducing the Bug

```python
from pydantic.color import Color

rgb = (0, 0, 1)
color = Color(rgb)
hsl_str = color.as_hsl()
color2 = Color(hsl_str)

print(f"Original: {color.as_rgb_tuple()}")
print(f"HSL: {hsl_str}")
print(f"After round-trip: {color2.as_rgb_tuple()}")
print(f"Match: {color == color2}")
```

Output:
```
Original: (0, 0, 1)
HSL: hsl(240, 100%, 0%)
After round-trip: (0, 0, 0)
Match: False
```

## Why This Is A Bug

The color `(0, 0, 1)` represents a very dark blue. When converted to HSL format, the lightness should be approximately 0.2%, but the `as_hsl()` method formats it with 0 decimal places (`:0.0%`), rounding it down to 0%. When this string is parsed back, it becomes pure black `(0, 0, 0)`, losing the blue component.

This violates the fundamental round-trip property that serialization and deserialization should preserve data. Users expect `Color(color.as_hsl()) == color` to always hold true.

## Fix

```diff
--- a/pydantic/color.py
+++ b/pydantic/color.py
@@ -193,9 +193,9 @@ class Color(_repr.Representation):
     def as_hsl(self) -> str:
         """Color as an `hsl(<h>, <s>, <l>)` or `hsl(<h>, <s>, <l>, <a>)` string."""
         if self._rgba.alpha is None:
             h, s, li = self.as_hsl_tuple(alpha=False)  # type: ignore
-            return f'hsl({h * 360:0.0f}, {s:0.0%}, {li:0.0%})'
+            return f'hsl({h * 360:0.1f}, {s:0.1%}, {li:0.1%})'
         else:
             h, s, li, a = self.as_hsl_tuple(alpha=True)  # type: ignore
-            return f'hsl({h * 360:0.0f}, {s:0.0%}, {li:0.0%}, {round(a, 2)})'
+            return f'hsl({h * 360:0.1f}, {s:0.1%}, {li:0.1%}, {round(a, 2)})'
```