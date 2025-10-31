# Bug Report: pydantic.color.Color RGBA Alpha Precision Loss

**Target**: `pydantic.color.Color.as_rgb()`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Color.as_rgb()` method rounds alpha values to 2 decimal places, causing precision loss and violating the round-trip property `Color(color.as_rgb()) == color` for colors with alpha values that don't round cleanly to 2 decimal places.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pydantic.color import Color

rgb_values = st.integers(min_value=0, max_value=255)
alpha_values = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

@given(st.tuples(rgb_values, rgb_values, rgb_values, alpha_values))
@settings(max_examples=1000)
def test_rgba_string_round_trip(rgba):
    color = Color(rgba)
    rgba_str = color.as_rgb()
    color2 = Color(rgba_str)
    assert color == color2
```

**Failing input**: `rgba=(0, 0, 0, 0.625)`

## Reproducing the Bug

```python
from pydantic.color import Color

rgba = (0, 0, 0, 0.625)
color = Color(rgba)
rgba_str = color.as_rgb()
color2 = Color(rgba_str)

print(f"Original: {color.as_rgb_tuple(alpha=True)}")
print(f"RGBA string: {rgba_str}")
print(f"After round-trip: {color2.as_rgb_tuple(alpha=True)}")
print(f"Match: {color == color2}")
```

Output:
```
Original: (0, 0, 0, 0.625)
RGBA string: rgba(0, 0, 0, 0.62)
After round-trip: (0, 0, 0, 0.62)
Match: False
```

## Why This Is A Bug

The `as_rgb()` method arbitrarily rounds alpha to 2 decimal places (line 163: `round(self._alpha_float(), 2)`), but the parser `parse_float_alpha()` can handle full floating-point precision. This asymmetry violates the round-trip property.

While CSS typically displays alpha with limited precision for readability, there's no technical reason to lose precision in the serialized format. Users who store and retrieve colors with specific alpha values will experience silent data corruption.

## Fix

```diff
--- a/pydantic/color.py
+++ b/pydantic/color.py
@@ -160,7 +160,7 @@ class Color(_repr.Representation):
         else:
             return (
                 f'rgba({float_to_255(self._rgba.r)}, {float_to_255(self._rgba.g)}, {float_to_255(self._rgba.b)}, '
-                f'{round(self._alpha_float(), 2)})'
+                f'{self._alpha_float()})'
             )
```

Alternatively, if preserving the 2-decimal display format is desired for compatibility, the fix could round more carefully to avoid precision loss where possible, or document that `as_rgb()` is lossy and provide a lossless alternative.