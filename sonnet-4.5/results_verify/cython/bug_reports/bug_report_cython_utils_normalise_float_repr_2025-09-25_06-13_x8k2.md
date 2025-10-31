# Bug Report: Cython.Utils.normalise_float_repr Fails on Negative Numbers

**Target**: `Cython.Utils.normalise_float_repr`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `normalise_float_repr` function in `Cython.Utils` fails to handle negative numbers correctly, producing either invalid output that cannot be parsed back to a float, or incorrect values that don't match the input.

## Property-Based Test

```python
import sys
from hypothesis import assume, given, strategies as st

sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Utils import normalise_float_repr


@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e50, max_value=1e50))
def test_normalise_float_repr_preserves_value(f):
    assume(f != 0.0 or str(f) != '-0.0')

    float_str = str(f)
    normalised = normalise_float_repr(float_str)

    original_value = float(float_str)
    normalised_value = float(normalised)

    assert original_value == normalised_value
```

**Failing inputs**:
- `f=-1.938987928904224e-24` → produces `'.0000000000000000000000-1938987928904224'` (invalid)
- `f=1.67660926681519e-08` → produces incorrect value `16766092.000000067` instead of `1.67660926681519e-08`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Utils import normalise_float_repr

result = normalise_float_repr("-1.938987928904224e-24")
print(f"Result: {result!r}")

float(result)
```

Output:
```
Result: '.0000000000000000000000-1938987928904224'
Traceback (most recent call last):
  ...
ValueError: could not convert string to float: '.0000000000000000000000-1938987928904224'
```

## Why This Is A Bug

The function claims to "Generate a 'normalised', simple digits string representation of a float value to allow string comparisons" (line 661-663 of Utils.py). However, it produces invalid output for negative numbers because:

1. The minus sign is never extracted or handled separately
2. When the decimal point is removed and digits are rearranged based on the exponent, the minus sign ends up in the wrong position
3. The result cannot be converted back to a float, violating the implicit contract that the output represents the same value

This is particularly problematic because:
- The function is used for string comparisons of float values
- The existing test suite (TestCythonUtils.py:169-202) only tests positive numbers
- Users would reasonably expect it to work with negative numbers

## Fix

```diff
--- a/Utils.py
+++ b/Utils.py
@@ -662,7 +662,12 @@ def normalise_float_repr(float_str):
     Generate a 'normalised', simple digits string representation of a float value
     to allow string comparisons.  Examples: '.123', '123.456', '123.'
     """
-    str_value = float_str.lower().lstrip('0')
+    # Handle negative numbers by extracting the sign
+    sign = ''
+    str_value = float_str.lower()
+    if str_value.startswith('-'):
+        sign = '-'
+        str_value = str_value[1:]
+    str_value = str_value.lstrip('0')

     exp = 0
     if 'E' in str_value or 'e' in str_value:
@@ -676,7 +681,7 @@ def normalise_float_repr(float_str):
         num_int_digits = len(str_value)
     exp += num_int_digits

-    result = (
+    result = sign + (
         str_value[:exp]
         + '0' * (exp - len(str_value))
         + '.'