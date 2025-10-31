# Bug Report: CapacityLimiter total_tokens Type Validation

**Target**: `anyio._core._synchronization.CapacityLimiter.total_tokens` (setter)
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `CapacityLimiter.total_tokens` property setter rejects valid `float` values despite the property being typed as `float` in all type hints and documentation. The validation incorrectly restricts inputs to `int` or `math.inf`, causing a `TypeError` when setting valid float values like `5.5`.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/anyio_env')
from hypothesis import given, strategies as st
from anyio._core._synchronization import CapacityLimiterAdapter

@given(st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
def test_capacity_limiter_accepts_float_total_tokens(value):
    limiter = CapacityLimiterAdapter(total_tokens=10)
    limiter.total_tokens = value
    assert limiter.total_tokens == value
```

**Failing input**: Any non-integer float, e.g., `5.5`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/anyio_env')
from anyio._core._synchronization import CapacityLimiterAdapter

limiter = CapacityLimiterAdapter(total_tokens=10)
limiter.total_tokens = 5.5
```

Output:
```
TypeError: total_tokens must be an int or math.inf
```

## Why This Is A Bug

The property is typed as `float` everywhere:
- Line 512: `def total_tokens(self) -> float:`
- Line 527: `def total_tokens(self, value: float) -> None:`
- Line 494: `def __new__(cls, total_tokens: float) -> CapacityLimiter:`
- Line 613: `def __init__(self, total_tokens: float) -> None:`

But the setter validation at line 645 rejects non-integer floats:
```python
if not isinstance(value, int) and value is not math.inf:
    raise TypeError("total_tokens must be an int or math.inf")
```

This bug exists in both:
1. `CapacityLimiterAdapter` (_synchronization.py:645)
2. `CapacityLimiter` (_backends/_asyncio.py:1963)

Float values make semantic sense - e.g., `total_tokens=5.5` allows 5 concurrent borrowers since the code compares `len(self._borrowers) < self._total_tokens`.

## Fix

```diff
--- a/anyio/_core/_synchronization.py
+++ b/anyio/_core/_synchronization.py
@@ -642,7 +642,7 @@ class CapacityLimiterAdapter(CapacityLimiter):
     @total_tokens.setter
     def total_tokens(self, value: float) -> None:
-        if not isinstance(value, int) and value is not math.inf:
-            raise TypeError("total_tokens must be an int or math.inf")
+        if not isinstance(value, (int, float)) or (isinstance(value, float) and math.isnan(value)):
+            raise TypeError("total_tokens must be a number")
         elif value < 1:
             raise ValueError("total_tokens must be >= 1")

--- a/anyio/_backends/_asyncio.py
+++ b/anyio/_backends/_asyncio.py
@@ -1960,7 +1960,7 @@ class CapacityLimiter(BaseCapacityLimiter):
     @total_tokens.setter
     def total_tokens(self, value: float) -> None:
-        if not isinstance(value, int) and not math.isinf(value):
-            raise TypeError("total_tokens must be an int or math.inf")
+        if not isinstance(value, (int, float)) or (isinstance(value, float) and math.isnan(value)):
+            raise TypeError("total_tokens must be a number")
         if value < 1:
             raise ValueError("total_tokens must be >= 1")
```