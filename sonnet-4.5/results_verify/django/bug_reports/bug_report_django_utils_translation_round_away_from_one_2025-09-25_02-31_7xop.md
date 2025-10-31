# Bug Report: django.utils.translation.round_away_from_one Incorrect Rounding for Small Negative Numbers

**Target**: `django.utils.translation.round_away_from_one`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `round_away_from_one` function incorrectly handles very small negative numbers (close to zero) due to floating-point precision loss when converting to Decimal, returning 0 instead of -1.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import django.utils.translation as trans
import math

@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1.0))
@settings(max_examples=1000)
def test_round_away_from_one_less_than_one(value):
    result = trans.round_away_from_one(value)
    expected = math.floor(value)
    assert result == expected, f"For {value}, expected {expected}, got {result}"
```

**Failing input**: `-1e-100` (and any negative number with absolute value less than ~1e-15)

## Reproducing the Bug

```python
import django.utils.translation as trans
import math

value = -1e-100
result = trans.round_away_from_one(value)
expected = math.floor(value)

print(f"round_away_from_one({value}) = {result}")
print(f"Expected: {expected}")
print(f"Bug: {result} != {expected}")
```

Output:
```
round_away_from_one(-1e-100) = 0
Expected: -1
Bug: 0 != -1
```

## Why This Is A Bug

The function `round_away_from_one` is documented to round values away from 1:
- Values > 1 should round up (ceiling)
- Values < 1 should round down (floor)
- Value = 1 should stay at 1

For very small negative numbers like `-1e-100`, the correct behavior is `floor(-1e-100) = -1`, but the function returns `0`.

The root cause is in the implementation:
```python
def round_away_from_one(value):
    return int(Decimal(value - 1).quantize(Decimal("0"), rounding=ROUND_UP)) + 1
```

When `value = -1e-100`:
1. `value - 1 = -1.0` (floating-point precision loss: `-1e-100 - 1` becomes exactly `-1.0`)
2. `Decimal(-1.0)` = `Decimal('-1')`
3. Quantize with ROUND_UP gives `-1`
4. `int(-1) + 1 = 0`

But the correct result should be `-1`.

## Fix

The issue is that `Decimal(value - 1)` loses precision when constructing a Decimal from a float after the subtraction. A better approach is to use `Decimal.from_float()` or convert to Decimal before the arithmetic:

```diff
def round_away_from_one(value):
-    return int(Decimal(value - 1).quantize(Decimal("0"), rounding=ROUND_UP)) + 1
+    from decimal import Decimal, ROUND_UP
+    # Convert to Decimal first to preserve precision
+    d = Decimal.from_float(value)
+    result = (d - 1).quantize(Decimal("0"), rounding=ROUND_UP)
+    return int(result) + 1
```

Alternatively, avoid Decimal altogether and use `math.ceil()` for values > 1 and `math.floor()` for values < 1:

```diff
def round_away_from_one(value):
-    return int(Decimal(value - 1).quantize(Decimal("0"), rounding=ROUND_UP)) + 1
+    import math
+    if value > 1:
+        return math.ceil(value)
+    elif value < 1:
+        return math.floor(value)
+    else:
+        return 1
```