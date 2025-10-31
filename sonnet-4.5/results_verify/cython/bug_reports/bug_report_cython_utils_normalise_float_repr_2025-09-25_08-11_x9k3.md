# Bug Report: Cython.Utils.normalise_float_repr Incorrect Handling of Negative Exponents

**Target**: `Cython.Utils.normalise_float_repr`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `normalise_float_repr` function produces incorrect results when normalizing float strings with negative exponents and multiple significant digits. The function returns a value that is orders of magnitude different from the input.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Utils import normalise_float_repr

@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e100, max_value=1e100))
@settings(max_examples=1000)
def test_normalise_float_repr_round_trip(x):
    if x == 0.0:
        return

    float_str = str(x)
    normalized = normalise_float_repr(float_str)

    assert float(normalized) == float(float_str), (
        f"Round-trip failed: {float_str} -> {normalized} "
        f"({float(float_str)} != {float(normalized)})"
    )
```

**Failing input**: `1.192092896e-07`

## Reproducing the Bug

```python
from Cython.Utils import normalise_float_repr

test_input = '1.192092896e-07'
result = normalise_float_repr(test_input)

print(f"Input:    {test_input}")
print(f"Output:   {result}")
print(f"Expected: .0000001192092896")
print()
print(f"Input float value:  {float(test_input)}")
print(f"Output float value: {float(result)}")

assert float(test_input) == float(result)
```

**Output:**
```
Input:    1.192092896e-07
Output:   1192.000000092896
Expected: .0000001192092896

Input float value:  1.192092896e-07
Output float value: 1192.000000092896
AssertionError
```

## Why This Is A Bug

The function's docstring states it should "Generate a 'normalised', simple digits string representation of a float value to allow string comparisons." The test suite at line 198 of `Tests/TestCythonUtils.py` explicitly verifies that `float(result) == float(float_str)` must hold.

However, when normalizing `1.192092896e-07`:
- **Expected**: `.0000001192092896` (equals `1.192092896e-07`)
- **Actual**: `1192.000000092896` (equals `1192.000000092896`, completely wrong)

The bug occurs because when `exp` becomes negative after processing, the slice `str_value[:exp]` (e.g., `str_value[:-6]`) produces the wrong result. The code incorrectly assumes that negative `exp` values will select from the beginning of the string, but instead the negative index slices from the end.

## Fix

```diff
--- a/Cython/Utils.py
+++ b/Cython/Utils.py
@@ -676,11 +676,16 @@ def normalise_float_repr(float_str):
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
+        result = (
+            '.'
+            + '0' * -exp
+            + str_value
+        ).rstrip('0')
+    else:
+        result = (
+            str_value[:exp]
+            + '0' * max(0, exp - len(str_value))
+            + '.'
+            + str_value[exp:]
+        ).rstrip('0')

     return result if result != '.' else '.0'
```