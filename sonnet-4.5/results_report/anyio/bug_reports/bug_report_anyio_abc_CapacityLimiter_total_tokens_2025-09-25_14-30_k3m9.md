# Bug Report: CapacityLimiter Type Hint Violation

**Target**: `anyio.abc.CapacityLimiter` (specifically `anyio._core._synchronization.CapacityLimiterAdapter`)
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `CapacityLimiter.total_tokens` property is type-hinted as `float` but the setter rejects all float values except `math.inf`, only accepting integers. This violates the type contract established by the type hints.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from anyio._core._synchronization import CapacityLimiter


@given(st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=200)
def test_capacity_limiter_accepts_float_tokens(tokens):
    limiter = CapacityLimiter(tokens)
    assert limiter.total_tokens == tokens
```

**Failing input**: `tokens=1.0` (or any float value that is not `math.inf`)

## Reproducing the Bug

```python
import math
from anyio._core._synchronization import CapacityLimiter

limiter = CapacityLimiter(1.5)

limiter = CapacityLimiter(1)
limiter.total_tokens = 2.5
```

Running this code raises:
```
TypeError: total_tokens must be an int or math.inf
```

## Why This Is A Bug

The type annotations throughout the class explicitly declare `total_tokens` as `float`:

1. `__new__(cls, total_tokens: float)` (line 494)
2. `__init__(self, total_tokens: float)` (line 613)
3. `@property def total_tokens(self) -> float:` (line 637)
4. `@total_tokens.setter def total_tokens(self, value: float) -> None:` (line 644)

However, the setter validation (lines 645-646) rejects all float values except `math.inf`:

```python
if not isinstance(value, int) and value is not math.inf:
    raise TypeError("total_tokens must be an int or math.inf")
```

This makes the type hints misleading and breaks the API contract. Users relying on type hints would expect float values like `1.5` to be accepted, but they are rejected at runtime.

## Fix

The fix should either:
1. Accept float values as the type hints promise, or
2. Change the type hints to `int | float` where float means specifically `math.inf`

Option 1 is more intuitive and matches the type annotations:

```diff
--- a/anyio/_core/_synchronization.py
+++ b/anyio/_core/_synchronization.py
@@ -642,7 +642,7 @@ class CapacityLimiterAdapter(CapacityLimiter):

     @total_tokens.setter
     def total_tokens(self, value: float) -> None:
-        if not isinstance(value, int) and value is not math.inf:
+        if not isinstance(value, (int, float)):
             raise TypeError("total_tokens must be an int or math.inf")
         elif value < 1:
             raise ValueError("total_tokens must be >= 1")
```

Note: The check should also be updated to handle `math.isnan()` and `math.isinf()` for negative infinity if needed, but the core issue is accepting float values at all.