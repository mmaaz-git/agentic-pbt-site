# Bug Report: pydantic.color.Color RGBA Alpha Precision Loss During Round-Trip Serialization

**Target**: `pydantic.color.Color.as_rgb()`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Color.as_rgb()` method arbitrarily rounds alpha values to 2 decimal places, causing silent data corruption and violating the expected round-trip property where `Color(color.as_rgb()) == color` for colors with alpha values that don't round cleanly to 2 decimal places.

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
    assert color == color2, f"Round-trip failed for {rgba}: {color.as_rgb_tuple(alpha=True)} != {color2.as_rgb_tuple(alpha=True)}"

if __name__ == "__main__":
    test_rgba_string_round_trip()
```

<details>

<summary>
**Failing input**: `rgba=(0, 0, 0, 0.125)`
</summary>
```
/home/npc/pbt/agentic-pbt/worker_/15/hypo.py:10: PydanticDeprecatedSince20: The `Color` class is deprecated, use `pydantic_extra_types` instead. See https://docs.pydantic.dev/latest/api/pydantic_extra_types_color/. Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.10/migration/
  color = Color(rgba)
/home/npc/pbt/agentic-pbt/worker_/15/hypo.py:12: PydanticDeprecatedSince20: The `Color` class is deprecated, use `pydantic_extra_types` instead. See https://docs.pydantic.dev/latest/api/pydantic_extra_types_color/. Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.10/migration/
  color2 = Color(rgba_str)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 16, in <module>
    test_rgba_string_round_trip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 8, in test_rgba_string_round_trip
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 13, in test_rgba_string_round_trip
    assert color == color2, f"Round-trip failed for {rgba}: {color.as_rgb_tuple(alpha=True)} != {color2.as_rgb_tuple(alpha=True)}"
           ^^^^^^^^^^^^^^^
AssertionError: Round-trip failed for (0, 0, 0, 0.125): (0, 0, 0, 0.125) != (0, 0, 0, 0.12)
Falsifying example: test_rgba_string_round_trip(
    rgba=(0, 0, 0, 0.125),
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/pydantic/color.py:186
```
</details>

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

<details>

<summary>
Output showing precision loss
</summary>
```
/home/npc/pbt/agentic-pbt/worker_/15/repo.py:4: PydanticDeprecatedSince20: The `Color` class is deprecated, use `pydantic_extra_types` instead. See https://docs.pydantic.dev/latest/api/pydantic_extra_types_color/. Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.10/migration/
  color = Color(rgba)
/home/npc/pbt/agentic-pbt/worker_/15/repo.py:6: PydanticDeprecatedSince20: The `Color` class is deprecated, use `pydantic_extra_types` instead. See https://docs.pydantic.dev/latest/api/pydantic_extra_types_color/. Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.10/migration/
  color2 = Color(rgba_str)
Original: (0, 0, 0, 0.625)
RGBA string: rgba(0, 0, 0, 0.62)
After round-trip: (0, 0, 0, 0.62)
Match: False
```
</details>

## Why This Is A Bug

This violates expected behavior for several critical reasons:

1. **Asymmetric Serialization/Deserialization**: The parser function `parse_float_alpha()` (lines 381-408 in `/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages/pydantic/color.py`) accepts and preserves full floating-point precision when parsing alpha values from strings. However, the serializer `as_rgb()` (line 163) arbitrarily rounds to 2 decimal places using `round(self._alpha_float(), 2)`. This asymmetry breaks the fundamental expectation of round-trip serialization.

2. **Silent Data Corruption**: The rounding happens silently without any warning or documentation. Users storing and retrieving colors with specific alpha values experience data loss without being aware of it. For example, an alpha value of 0.625 becomes 0.62, losing 0.005 of precision.

3. **No Technical Justification**: The CSS3 Color Module Level 3 specification, which this module explicitly references in its documentation (lines 1-2), does not mandate or even suggest rounding alpha values to 2 decimal places. CSS supports arbitrary precision for alpha values.

4. **Undocumented Behavior**: The method's docstring (line 157) simply states "Color as an `rgb(<r>, <g>, <b>)` or `rgba(<r>, <g>, <b>, <a>)` string" with no mention of precision limitations or lossy operations.

5. **Internal Inconsistency**: The internal `RGBA` class stores alpha as a full-precision float, and other methods like `as_rgb_tuple()` return the full precision. Only the string serialization loses precision, creating an inconsistent API.

## Relevant Context

- **Deprecation Status**: The Color class is deprecated in Pydantic V2.0 and will be removed in V3.0. Users are directed to use `pydantic_extra_types` instead. However, this doesn't invalidate the bug for users still using this code.

- **Similar HSL Issue**: The `as_hsl()` method (line 199) also rounds alpha to 2 decimal places, suggesting this might be a systematic issue in the string serialization methods.

- **CSS Specification**: The CSS3 Color Module Level 3 (referenced at http://www.w3.org/TR/css3-color/#svg-color) does not specify precision requirements for alpha values. Modern browsers accept and preserve arbitrary decimal precision in RGBA values.

- **Code Location**: The problematic rounding occurs at line 163 in `/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages/pydantic/color.py`

## Proposed Fix

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

     def as_hsl(self) -> str:
@@ -196,7 +196,7 @@ class Color(_repr.Representation):
             return f'hsl({h * 360:0.0f}, {s:0.0%}, {li:0.0%})'
         else:
             h, s, li, a = self.as_hsl_tuple(alpha=True)  # type: ignore
-            return f'hsl({h * 360:0.0f}, {s:0.0%}, {li:0.0%}, {round(a, 2)})'
+            return f'hsl({h * 360:0.0f}, {s:0.0%}, {li:0.0%}, {a})'
```