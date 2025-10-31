# Bug Report: anyio.abc.CapacityLimiter total_tokens Type Contract Violation

**Target**: `anyio.abc.CapacityLimiter.total_tokens` setter
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `CapacityLimiter.total_tokens` setter is type-annotated to accept `float` values, but the implementation only accepts `int` or `math.inf`, incorrectly rejecting valid float values like `2.5`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import math
from anyio.abc import CapacityLimiter


@given(st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False).filter(lambda x: not x.is_integer()))
def test_capacity_limiter_accepts_float_tokens(value):
    limiter = CapacityLimiter(10.0)
    limiter.total_tokens = value
    assert limiter.total_tokens == value
```

**Failing input**: Any non-integer float value >= 1.0, e.g., `2.5`

## Reproducing the Bug

```python
import math
from anyio.abc import CapacityLimiter

limiter = CapacityLimiter(10.0)

limiter.total_tokens = 5
print(f"Integer works: {limiter.total_tokens}")

limiter.total_tokens = math.inf
print(f"math.inf works: {limiter.total_tokens}")

limiter.total_tokens = 2.5
```

**Output**:
```
Integer works: 5
math.inf works: inf
TypeError: total_tokens must be an int or math.inf
```

## Why This Is A Bug

The base class `CapacityLimiter` defines the property signature at `/lib/python3.13/site-packages/anyio/_core/_synchronization.py:527`:

```python
@total_tokens.setter
def total_tokens(self, value: float) -> None:
```

The implementation in `CapacityLimiterAdapter` at lines 644-648 violates this contract:

```python
@total_tokens.setter
def total_tokens(self, value: float) -> None:
    if not isinstance(value, int) and value is not math.inf:
        raise TypeError("total_tokens must be an int or math.inf")
    elif value < 1:
        raise ValueError("total_tokens must be >= 1")
```

The type annotation promises to accept `float` values, but the validation rejects all floats except those that are exactly `math.inf`. This creates a type contract violation where valid float values like `2.5` are incorrectly rejected.

Furthermore, the property getter returns `float` (line 512, 637) and `available_tokens` also returns `float` (line 536), suggesting that fractional token values are intentionally supported by the design.

## Fix

```diff
--- a/anyio/_core/_synchronization.py
+++ b/anyio/_core/_synchronization.py
@@ -642,7 +642,7 @@ class CapacityLimiterAdapter(CapacityLimiter):

     @total_tokens.setter
     def total_tokens(self, value: float) -> None:
-        if not isinstance(value, int) and value is not math.inf:
+        if not isinstance(value, (int, float)) or (isinstance(value, float) and math.isnan(value)):
             raise TypeError("total_tokens must be an int or math.inf")
         elif value < 1:
             raise ValueError("total_tokens must be >= 1")
```