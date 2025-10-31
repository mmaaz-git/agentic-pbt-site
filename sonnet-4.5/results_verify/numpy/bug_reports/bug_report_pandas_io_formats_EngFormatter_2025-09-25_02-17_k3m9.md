# Bug Report: EngFormatter Silently Loses Precision for Numbers Outside [-24, 24] Exponent Range

**Target**: `pandas.io.formats.format.EngFormatter`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `EngFormatter` class clamps exponents to the range [-24, 24] even when `use_eng_prefix=False`, causing silent data loss for very small or very large numbers. Numbers like 1e-50 are incorrectly formatted as "0.0E-24", losing all significant information.

## Property-Based Test

```python
from hypothesis import assume, given, settings, strategies as st
import pandas.io.formats.format as fmt


@given(
    num=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e50, max_value=1e50),
    accuracy=st.integers(min_value=1, max_value=10)
)
@settings(max_examples=500)
def test_engformatter_parse_roundtrip(num, accuracy):
    assume(num != 0)

    formatter = fmt.EngFormatter(accuracy=accuracy, use_eng_prefix=False)
    formatted = formatter(num)

    parsed = float(formatted)
    relative_error = abs(parsed - num) / abs(num) if num != 0 else 0

    max_expected_error = 10 ** (-accuracy + 1)
    assert relative_error < max_expected_error, \
        f"Round-trip error too large: {num} -> '{formatted}' -> {parsed}, error: {relative_error}"
```

**Failing input**: `num=2.844615157173927e-200, accuracy=1`

## Reproducing the Bug

```python
import pandas.io.formats.format as fmt

formatter = fmt.EngFormatter(accuracy=1, use_eng_prefix=False)

num = 1e-50
formatted = formatter(num)
parsed = float(formatted)

print(f"Original: {num}")
print(f"Formatted: '{formatted}'")
print(f"Parsed back: {parsed}")
```

Output:
```
Original: 1e-50
Formatted: ' 0.0E-24'
Parsed back: 0.0
```

## Why This Is A Bug

The `EngFormatter.__call__` method unconditionally clamps the exponent to the range of `ENG_PREFIXES` ([-24, 24]) on lines:

```python
pow10 = pow10.min(max(self.ENG_PREFIXES.keys()))  # Clamps to 24
pow10 = pow10.max(min(self.ENG_PREFIXES.keys()))  # Clamps to -24
```

When `use_eng_prefix=False`, the formatter should use E notation (like "1.0E-50") which can represent any exponent, not just those with engineering prefixes. The clamping makes sense only when `use_eng_prefix=True`, where we need a prefix letter from the dictionary.

This violates expected behavior because:
1. The formatter silently loses information without warning
2. When `use_eng_prefix=False`, there's no reason to limit the exponent range
3. The round-trip property (format then parse) fails for numbers outside [-1e24, 1e24]

## Fix

```diff
--- a/pandas/io/formats/format.py
+++ b/pandas/io/formats/format.py
@@ -1234,8 +1234,12 @@ class EngFormatter:
         else:
             pow10 = Decimal(0)

-        pow10 = pow10.min(max(self.ENG_PREFIXES.keys()))
-        pow10 = pow10.max(min(self.ENG_PREFIXES.keys()))
+        if self.use_eng_prefix:
+            # Only clamp when using prefixes, since we need a valid prefix key
+            pow10 = pow10.min(max(self.ENG_PREFIXES.keys()))
+            pow10 = pow10.max(min(self.ENG_PREFIXES.keys()))
+        # else: Allow any exponent when using E notation
+
         int_pow10 = int(pow10)

         if self.use_eng_prefix:
```