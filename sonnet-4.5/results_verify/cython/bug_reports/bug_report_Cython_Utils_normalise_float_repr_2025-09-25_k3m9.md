# Bug Report: Cython.Utils.normalise_float_repr Incorrect Handling of Negative Exponents

**Target**: `Cython.Utils.normalise_float_repr`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `normalise_float_repr()` function incorrectly handles scientific notation with negative exponents, producing results that are off by many orders of magnitude. For example, `6.103515625e-05` (≈0.00006104) is incorrectly normalized to `'610351.00005625'` (≈610351).

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from Cython.Utils import normalise_float_repr

@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e100, max_value=1e100))
def test_normalise_float_repr_preserves_value(f):
    float_str = str(f)
    normalized = normalise_float_repr(float_str)

    original_value = float(float_str)
    normalized_value = float(normalized)

    assert original_value == normalized_value
```

**Failing input**: `f=6.103515625e-05`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Utils import normalise_float_repr

float_str = '6.103515625e-05'
normalized = normalise_float_repr(float_str)

print(f"Input: {float_str}")
print(f"Normalized: {normalized}")
print(f"Original float: {float(float_str)}")
print(f"Normalized float: {float(normalized)}")

assert float(float_str) == float(normalized)
```

**Output:**
```
Input: 6.103515625e-05
Normalized: 610351.00005625
Original float: 6.103515625e-05
Normalized float: 610351.00005625
AssertionError
```

## Why This Is A Bug

The function's docstring states it should "Generate a 'normalised', simple digits string representation of a float value to allow string comparisons" and the existing test suite verifies that `float(normalise_float_repr(x)) == float(x)`. However, when the exponent is negative (numbers < 1 in scientific notation), the slicing logic at lines 679-685 is fundamentally broken.

For `'6.103515625e-05'`, after parsing:
- `str_value = '6103515625'` (decimal point removed)
- `exp = -4` (original -5, plus 1 for the single digit before decimal)

The code then does:
```python
str_value[:exp]  # str_value[:-4] = '610351' ← WRONG! Should be '' for negative exp
```

This puts digits before the decimal point when they should all be after it with leading zeros.

## Fix

```diff
--- a/Cython/Utils.py
+++ b/Cython/Utils.py
@@ -676,12 +676,17 @@ def normalise_float_repr(float_str):
         num_int_digits = len(str_value)
     exp += num_int_digits

-    result = (
-        str_value[:exp]
-        + '0' * (exp - len(str_value))
-        + '.'
-        + '0' * -exp
-        + str_value[exp:]
-    ).rstrip('0')
+    if exp <= 0:
+        result = ('.' + '0' * -exp + str_value).rstrip('0')
+    else:
+        result = (
+            str_value[:exp]
+            + '0' * (exp - len(str_value))
+            + '.'
+            + str_value[exp:]
+        ).rstrip('0')

     return result if result != '.' else '.0'
```