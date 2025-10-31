# Bug Report: anyio.CapacityLimiter Type Annotation Mismatch Causes Runtime Errors

**Target**: `anyio.CapacityLimiter` and `anyio._core._synchronization.CapacityLimiterAdapter`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `CapacityLimiter` class has a type annotation of `float` for its `total_tokens` parameter but only accepts `int` or `math.inf` at runtime, causing a contract violation between static type checking and runtime validation.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import math
from anyio import CapacityLimiter

@given(st.floats(min_value=1.0, max_value=1000.0).filter(
    lambda x: x != math.inf and not (isinstance(x, float) and x == int(x)) and not math.isnan(x)
))
def test_capacity_limiter_type_contract(total_tokens):
    """
    Test that CapacityLimiter accepts all non-negative floats >= 1 as per type annotation.
    The type annotation says 'float', so all floats >= 1 should be valid.
    """
    limiter = CapacityLimiter(total_tokens)
    assert limiter.total_tokens == total_tokens

if __name__ == "__main__":
    test_capacity_limiter_type_contract()
```

<details>

<summary>
**Failing input**: `total_tokens=1.5`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/site-packages/anyio/_core/_synchronization.py", line 496, in __new__
    return get_async_backend().create_capacity_limiter(total_tokens)
           ~~~~~~~~~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/anyio/_core/_eventloop.py", line 156, in get_async_backend
    asynclib_name = sniffio.current_async_library()
  File "/home/npc/miniconda/lib/python3.13/site-packages/sniffio/_impl.py", line 93, in current_async_library
    raise AsyncLibraryNotFoundError(
        "unknown async library, or not in async context"
    )
sniffio._impl.AsyncLibraryNotFoundError: unknown async library, or not in async context

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/0/hypo.py", line 17, in <module>
    test_capacity_limiter_type_contract()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/0/hypo.py", line 6, in test_capacity_limiter_type_contract
    lambda x: x != math.inf and not (isinstance(x, float) and x == int(x)) and not math.isnan(x)
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/0/hypo.py", line 13, in test_capacity_limiter_type_contract
    limiter = CapacityLimiter(total_tokens)
  File "/home/npc/miniconda/lib/python3.13/site-packages/anyio/_core/_synchronization.py", line 498, in __new__
    return CapacityLimiterAdapter(total_tokens)
  File "/home/npc/miniconda/lib/python3.13/site-packages/anyio/_core/_synchronization.py", line 614, in __init__
    self.total_tokens = total_tokens
    ^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/anyio/_core/_synchronization.py", line 646, in total_tokens
    raise TypeError("total_tokens must be an int or math.inf")
TypeError: total_tokens must be an int or math.inf
Falsifying example: test_capacity_limiter_type_contract(
    total_tokens=1.5,
)
```
</details>

## Reproducing the Bug

```python
from anyio import CapacityLimiter
import math

print("Test 1: integer value")
try:
    limiter = CapacityLimiter(5)
    print(f"✓ Accepted 5 (int): total_tokens={limiter.total_tokens}")
except Exception as e:
    print(f"✗ Rejected 5: {e}")

print("\nTest 2: math.inf")
try:
    limiter = CapacityLimiter(math.inf)
    print(f"✓ Accepted math.inf: total_tokens={limiter.total_tokens}")
except Exception as e:
    print(f"✗ Rejected math.inf: {e}")

print("\nTest 3: float 5.5")
try:
    limiter = CapacityLimiter(5.5)
    print(f"✓ Accepted 5.5 (float): total_tokens={limiter.total_tokens}")
except TypeError as e:
    print(f"✗ Rejected 5.5: {e}")

print("\nTest 4: float 1.5")
try:
    limiter = CapacityLimiter(1.5)
    print(f"✓ Accepted 1.5 (float): total_tokens={limiter.total_tokens}")
except TypeError as e:
    print(f"✗ Rejected 1.5: {e}")

print("\nTest 5: float 10.25")
try:
    limiter = CapacityLimiter(10.25)
    print(f"✓ Accepted 10.25 (float): total_tokens={limiter.total_tokens}")
except TypeError as e:
    print(f"✗ Rejected 10.25: {e}")
```

<details>

<summary>
TypeError raised for valid float values like 1.5, 5.5, and 10.25
</summary>
```
Test 1: integer value
✓ Accepted 5 (int): total_tokens=5

Test 2: math.inf
✓ Accepted math.inf: total_tokens=inf

Test 3: float 5.5
✗ Rejected 5.5: total_tokens must be an int or math.inf

Test 4: float 1.5
✗ Rejected 1.5: total_tokens must be an int or math.inf

Test 5: float 10.25
✗ Rejected 10.25: total_tokens must be an int or math.inf
```
</details>

## Why This Is A Bug

This violates the explicit type contract established by the type annotations. The type system promises to accept any `float` value, but the runtime validation rejects non-integer floats except for `math.inf`. This creates a serious discrepancy where:

1. **Type annotations lie**: The `CapacityLimiter.__new__` method (line 494) and `CapacityLimiterAdapter.__init__` method (line 613) both declare `total_tokens: float`, explicitly stating that any float value should be accepted.

2. **Runtime validation contradicts type hints**: The `total_tokens` setter (lines 644-649) validates with `if not isinstance(value, int) and value is not math.inf:`, rejecting valid float values like 1.5, 2.5, or 10.25.

3. **Static analysis tools will miss errors**: Code that passes mypy or pyright type checking will fail at runtime. For example:
   ```python
   capacity: float = calculate_capacity()  # returns 1.5
   limiter = CapacityLimiter(capacity)  # Type checker: ✓ Runtime: ✗
   ```

4. **Inconsistent identity check**: Line 645 uses `value is not math.inf` instead of the more appropriate `value != math.inf` for value comparison. While this works for `math.inf` (a singleton), it's fragile and relies on implementation details.

## Relevant Context

The anyio library is widely used for asynchronous programming in Python, and `CapacityLimiter` is a key synchronization primitive for rate limiting and resource management. The type annotation issue affects:

- Users who rely on type hints for API documentation
- Projects using strict type checking with mypy or pyright
- Code generation tools that depend on accurate type annotations
- IDE autocompletion and type inference

The bug is located in `/site-packages/anyio/_core/_synchronization.py`. The `CapacityLimiterAdapter` is used as a fallback when no async backend is available (as seen when running outside an async context).

Documentation reference: The docstring for `total_tokens` property (lines 513-523) doesn't explicitly state the int-only requirement, only mentioning it's "The total number of tokens available for borrowing."

## Proposed Fix

Update the type annotations to accurately reflect the runtime validation requirements:

```diff
--- a/anyio/_core/_synchronization.py
+++ b/anyio/_core/_synchronization.py
@@ -491,7 +491,7 @@


 class CapacityLimiter:
-    def __new__(cls, total_tokens: float) -> CapacityLimiter:
+    def __new__(cls, total_tokens: int | float) -> CapacityLimiter:
         try:
             return get_async_backend().create_capacity_limiter(total_tokens)
         except AsyncLibraryNotFoundError:
@@ -610,7 +610,7 @@ class CapacityLimiterAdapter(CapacityLimiter):
     def __new__(cls, total_tokens: float) -> CapacityLimiterAdapter:
         return object.__new__(cls)

-    def __init__(self, total_tokens: float) -> None:
+    def __init__(self, total_tokens: int | float) -> None:
         self.total_tokens = total_tokens

     @property
@@ -642,7 +642,7 @@ class CapacityLimiterAdapter(CapacityLimiter):

     @total_tokens.setter
     def total_tokens(self, value: float) -> None:
-        if not isinstance(value, int) and value is not math.inf:
+        if not isinstance(value, int) and value != math.inf:
             raise TypeError("total_tokens must be an int or math.inf")
         elif value < 1:
             raise ValueError("total_tokens must be >= 1")
```