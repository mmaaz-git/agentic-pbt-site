# Bug Report: pydantic.color Color HSL Round-Trip Precision Loss

**Target**: `pydantic.color.Color.as_hsl`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Color.as_hsl()` method loses precision when converting colors to HSL string format, causing RGB values to change by 1-2 units when round-tripping through HSL strings.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pydantic.color import Color

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

    assert all(abs(a - b) <= 1 for a, b in zip(t1, t2)), \
        f"Round-trip failed: {t1} -> {hsl_str} -> {t2}"
```

**Failing input**: `(0, 0, 22)`

## Reproducing the Bug

```python
from pydantic.color import Color

color = Color((0, 0, 22))
print(f"Original RGB: {color.as_rgb_tuple()}")

hsl_str = color.as_hsl()
print(f"HSL string: {hsl_str}")

color2 = Color(hsl_str)
print(f"After round-trip: {color2.as_rgb_tuple()}")

print(f"Difference: {abs(22 - color2.as_rgb_tuple()[2])}")
```

Output:
```
Original RGB: (0, 0, 22)
HSL string: hsl(240, 100%, 4%)
After round-trip: (0, 0, 20)
Difference: 2
```

Additional examples:
- `(0, 0, 1)` → `hsl(240, 100%, 0%)` → `(0, 0, 0)` (diff: 1)
- `(0, 0, 2)` → `hsl(240, 100%, 0%)` → `(0, 0, 0)` (diff: 2)
- `(0, 0, 22)` → `hsl(240, 100%, 4%)` → `(0, 0, 20)` (diff: 2)

## Why This Is A Bug

The `as_hsl()` method at line 196 uses format string `{li:0.0%}` which rounds percentages to integers:

```python
return f'hsl({h * 360:0.0f}, {s:0.0%}, {li:0.0%})'
```

This means:
- 4.3% rounds to 4%
- 4.8% rounds to 5%

When parsing the HSL string back, "4%" represents a range of values (3.5% to 4.5%), causing precision loss. Colors that map to different percentages before rounding (like 4.3% and 4.0%) will produce the same HSL string but different RGB values when parsed back.

While HSL is typically used for display rather than exact serialization, the Color class provides both serialization (`as_hsl()`) and parsing (`Color(hsl_str)`) without documenting precision limitations, leading users to expect round-tripping to work.

## Fix

Use one decimal place for percentages to reduce precision loss:

```diff
--- a/pydantic/color.py
+++ b/pydantic/color.py
@@ -193,9 +193,9 @@ class Color(_repr.Representation):
     def as_hsl(self) -> str:
         """Color as an `hsl(<h>, <s>, <l>)` or `hsl(<h>, <s>, <l>, <a>)` string."""
         if self._rgba.alpha is None:
             h, s, li = self.as_hsl_tuple(alpha=False)  # type: ignore
-            return f'hsl({h * 360:0.0f}, {s:0.0%}, {li:0.0%})'
+            return f'hsl({h * 360:0.0f}, {s:0.1%}, {li:0.1%})'
         else:
             h, s, li, a = self.as_hsl_tuple(alpha=True)  # type: ignore
-            return f'hsl({h * 360:0.0f}, {s:0.0%}, {li:0.0%}, {round(a, 2)})'
+            return f'hsl({h * 360:0.0f}, {s:0.1%}, {li:0.1%}, {round(a, 2)})'
```

Note: The Color class is deprecated in Pydantic v2.0, but this bug affects users still on v2.x before migrating to `pydantic_extra_types`.