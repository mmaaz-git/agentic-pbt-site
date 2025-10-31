# Bug Report: anyio.CapacityLimiter Rejects float('inf')

**Target**: `anyio.CapacityLimiter` (specifically `CapacityLimiterAdapter.total_tokens` setter)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`CapacityLimiter` uses identity comparison (`is`) instead of equality comparison (`==`) when checking for infinity, causing it to reject `float('inf')` even though `float('inf') == math.inf`.

## Property-Based Test

```python
from anyio import CapacityLimiter
from hypothesis import given, strategies as st
import math

@given(st.sampled_from([math.inf, float('inf'), float('infinity')]))
def test_capacity_limiter_accepts_all_infinity_representations(inf_value):
    limiter = CapacityLimiter(inf_value)
    assert limiter.total_tokens == inf_value
```

**Failing input**: `float('inf')` or `float('infinity')`

## Reproducing the Bug

```python
import math
from anyio import CapacityLimiter

print(f"float('inf') == math.inf: {float('inf') == math.inf}")
print(f"float('inf') is math.inf: {float('inf') is math.inf}")

try:
    limiter = CapacityLimiter(float('inf'))
    print("Success")
except TypeError as e:
    print(f"Bug: {e}")
```

Output:
```
float('inf') == math.inf: True
float('inf') is math.inf: False
Bug: total_tokens must be an int or math.inf
```

## Why This Is A Bug

In Python, `float('inf')`, `float('infinity')`, and `math.inf` all represent the same mathematical value (positive infinity) and compare equal. Users may create infinity values through various means (parsing strings, calculations, etc.) that produce `float('inf')` rather than the specific `math.inf` object.

The validation in `anyio/_core/_synchronization.py:645` uses identity comparison:

```python
if not isinstance(value, int) and value is not math.inf:
    raise TypeError("total_tokens must be an int or math.inf")
```

This rejects semantically valid infinity values that aren't the exact `math.inf` object. The documented behavior says "must be an int or math.inf" but users would reasonably expect any infinity value to work.

## Fix

```diff
--- a/anyio/_core/_synchronization.py
+++ b/anyio/_core/_synchronization.py
@@ -642,7 +642,7 @@ class CapacityLimiterAdapter(CapacityLimiter):

     @total_tokens.setter
     def total_tokens(self, value: float) -> None:
-        if not isinstance(value, int) and value is not math.inf:
+        if not isinstance(value, int) and value != math.inf:
             raise TypeError("total_tokens must be an int or math.inf")
         elif value < 1:
             raise ValueError("total_tokens must be >= 1")
```

Alternatively, check for infinity using `math.isinf()`:

```diff
-        if not isinstance(value, int) and value is not math.inf:
+        if not isinstance(value, int) and not math.isinf(value):
             raise TypeError("total_tokens must be an int or math.inf")
```

Note: The second approach would also accept negative infinity, which may or may not be desired. The simpler `!= math.inf` fix is more conservative and matches the current intent.