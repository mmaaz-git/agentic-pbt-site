# Bug Report: anyio CapacityLimiter.total_tokens Type Contract Violation

**Target**: `anyio.abc.CapacityLimiter.total_tokens`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `CapacityLimiter.total_tokens` setter violates its type contract by rejecting valid float values despite being annotated to accept `float`, causing runtime errors for type-checked code.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import math
from anyio.abc import CapacityLimiter


@given(st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False).filter(lambda x: not x.is_integer()))
def test_capacity_limiter_accepts_float_tokens(value):
    limiter = CapacityLimiter(10)  # Use integer for initialization to avoid early failure
    limiter.total_tokens = value
    assert limiter.total_tokens == value

if __name__ == "__main__":
    test_capacity_limiter_accepts_float_tokens()
```

<details>

<summary>
**Failing input**: `value=1.5`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 13, in <module>
    test_capacity_limiter_accepts_float_tokens()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 7, in test_capacity_limiter_accepts_float_tokens
    def test_capacity_limiter_accepts_float_tokens(value):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 9, in test_capacity_limiter_accepts_float_tokens
    limiter.total_tokens = value
    ^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/anyio/_core/_synchronization.py", line 646, in total_tokens
    raise TypeError("total_tokens must be an int or math.inf")
TypeError: total_tokens must be an int or math.inf
Falsifying example: test_capacity_limiter_accepts_float_tokens(
    value=1.5,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import math
from anyio.abc import CapacityLimiter

limiter = CapacityLimiter(10)  # Use integer for initialization

limiter.total_tokens = 5
print(f"Integer works: {limiter.total_tokens}")

limiter.total_tokens = math.inf
print(f"math.inf works: {limiter.total_tokens}")

limiter.total_tokens = 2.5
print(f"Float should work but fails: {limiter.total_tokens}")
```

<details>

<summary>
TypeError when setting float value
</summary>
```
Integer works: 5
math.inf works: inf
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/54/repo.py", line 12, in <module>
    limiter.total_tokens = 2.5
    ^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/anyio/_core/_synchronization.py", line 646, in total_tokens
    raise TypeError("total_tokens must be an int or math.inf")
TypeError: total_tokens must be an int or math.inf
```
</details>

## Why This Is A Bug

The implementation violates its own type contract. The base class `CapacityLimiter` defines the property signature with type annotation `float`:

- At line 494: `def __new__(cls, total_tokens: float) -> CapacityLimiter`
- At line 527: `def total_tokens(self, value: float) -> None`
- At line 512: `def total_tokens(self) -> float` (getter returns float)
- At line 536: `def available_tokens(self) -> float` (related property also returns float)

However, the `CapacityLimiterAdapter` implementation at lines 644-646 incorrectly restricts the setter:

```python
@total_tokens.setter
def total_tokens(self, value: float) -> None:
    if not isinstance(value, int) and value is not math.inf:
        raise TypeError("total_tokens must be an int or math.inf")
```

This creates a type contract violation where:
1. Type checkers (mypy, pyright) will accept code passing float values based on the annotation
2. That same code will fail at runtime with a TypeError
3. The restriction is inconsistent - it accepts `math.inf` (a float) but rejects `2.5` (also a float)

## Relevant Context

The CapacityLimiter is used for rate limiting and resource management in async applications. The ability to use fractional tokens could be valuable for:
- Weighted resource allocation (e.g., 1.5 tokens for higher-priority tasks)
- Fine-grained rate limiting (e.g., 2.5 requests per second)
- Proportional resource sharing between different services

The documentation (version 3.0 changelog) states that `total_tokens` became writable but doesn't mention any restriction to integers only. All type annotations consistently use `float`, suggesting fractional tokens are an intended feature.

Source code location: `/lib/python3.13/site-packages/anyio/_core/_synchronization.py`
- Base class definition: lines 493-604
- Problematic implementation: lines 607-697, specifically line 645

## Proposed Fix

```diff
--- a/anyio/_core/_synchronization.py
+++ b/anyio/_core/_synchronization.py
@@ -642,7 +642,7 @@ class CapacityLimiterAdapter(CapacityLimiter):

     @total_tokens.setter
     def total_tokens(self, value: float) -> None:
-        if not isinstance(value, int) and value is not math.inf:
-            raise TypeError("total_tokens must be an int or math.inf")
+        if not isinstance(value, (int, float)) or (isinstance(value, float) and math.isnan(value)):
+            raise TypeError("total_tokens must be a number (not NaN)")
         elif value < 1:
             raise ValueError("total_tokens must be >= 1")
```