# Bug Report: anyio.abc.CapacityLimiter Type Contract Violation for Float Values

**Target**: `anyio.abc.CapacityLimiter` (specifically `anyio._core._synchronization.CapacityLimiterAdapter`)
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `CapacityLimiter` class type hints declare `total_tokens` as `float` throughout the API, but the implementation rejects all float values except `math.inf`, only accepting integers. This violates the type contract and misleads developers using type checkers, IDE autocomplete, or reading the type annotations.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from anyio._core._synchronization import CapacityLimiter


@given(st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=200)
def test_capacity_limiter_accepts_float_tokens(tokens):
    limiter = CapacityLimiter(tokens)
    assert limiter.total_tokens == tokens


if __name__ == "__main__":
    test_capacity_limiter_accepts_float_tokens()
```

<details>

<summary>
**Failing input**: `tokens=1.0`
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
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 13, in <module>
    test_capacity_limiter_accepts_float_tokens()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 6, in test_capacity_limiter_accepts_float_tokens
    @settings(max_examples=200)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 8, in test_capacity_limiter_accepts_float_tokens
    limiter = CapacityLimiter(tokens)
  File "/home/npc/miniconda/lib/python3.13/site-packages/anyio/_core/_synchronization.py", line 498, in __new__
    return CapacityLimiterAdapter(total_tokens)
  File "/home/npc/miniconda/lib/python3.13/site-packages/anyio/_core/_synchronization.py", line 614, in __init__
    self.total_tokens = total_tokens
    ^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/anyio/_core/_synchronization.py", line 646, in total_tokens
    raise TypeError("total_tokens must be an int or math.inf")
TypeError: total_tokens must be an int or math.inf
Falsifying example: test_capacity_limiter_accepts_float_tokens(
    tokens=1.0,
)
```
</details>

## Reproducing the Bug

```python
import math
from anyio._core._synchronization import CapacityLimiter

print("Test 1: Creating CapacityLimiter with float 1.5")
try:
    limiter = CapacityLimiter(1.5)
    print(f"Success: Created limiter with total_tokens={limiter.total_tokens}")
except TypeError as e:
    print(f"Failed with TypeError: {e}")

print("\nTest 2: Creating CapacityLimiter with float 1.0")
try:
    limiter = CapacityLimiter(1.0)
    print(f"Success: Created limiter with total_tokens={limiter.total_tokens}")
except TypeError as e:
    print(f"Failed with TypeError: {e}")

print("\nTest 3: Creating CapacityLimiter with int 1")
try:
    limiter = CapacityLimiter(1)
    print(f"Success: Created limiter with total_tokens={limiter.total_tokens}")
except TypeError as e:
    print(f"Failed with TypeError: {e}")

print("\nTest 4: Setting total_tokens to float 2.5")
try:
    limiter = CapacityLimiter(1)
    limiter.total_tokens = 2.5
    print(f"Success: Set total_tokens to {limiter.total_tokens}")
except TypeError as e:
    print(f"Failed with TypeError: {e}")

print("\nTest 5: Setting total_tokens to math.inf")
try:
    limiter = CapacityLimiter(1)
    limiter.total_tokens = math.inf
    print(f"Success: Set total_tokens to {limiter.total_tokens}")
except TypeError as e:
    print(f"Failed with TypeError: {e}")
```

<details>

<summary>
TypeError raised for all float values except math.inf
</summary>
```
Test 1: Creating CapacityLimiter with float 1.5
Failed with TypeError: total_tokens must be an int or math.inf

Test 2: Creating CapacityLimiter with float 1.0
Failed with TypeError: total_tokens must be an int or math.inf

Test 3: Creating CapacityLimiter with int 1
Success: Created limiter with total_tokens=1

Test 4: Setting total_tokens to float 2.5
Failed with TypeError: total_tokens must be an int or math.inf

Test 5: Setting total_tokens to math.inf
Success: Set total_tokens to inf
```
</details>

## Why This Is A Bug

This is a clear violation of the type contract established by the API's type hints. Throughout the `CapacityLimiter` class hierarchy, `total_tokens` is consistently typed as `float`:

1. **Base class declaration** (`_synchronization.py:494`): `def __new__(cls, total_tokens: float) -> CapacityLimiter:`
2. **Property getter** (`_synchronization.py:512,637`): `def total_tokens(self) -> float:`
3. **Property setter** (`_synchronization.py:527,644`): `def total_tokens(self, value: float) -> None:`
4. **Adapter initialization** (`_synchronization.py:613`): `def __init__(self, total_tokens: float) -> None:`

However, the implementation in `CapacityLimiterAdapter.total_tokens` setter (`_synchronization.py:645-646`) explicitly rejects float values:

```python
if not isinstance(value, int) and value is not math.inf:
    raise TypeError("total_tokens must be an int or math.inf")
```

This creates multiple problems:

1. **Type checker confusion**: Tools like mypy, pyright, and IDE type checkers will accept code passing float values, but it will fail at runtime
2. **Developer experience**: IDE autocomplete suggests float is acceptable based on type hints
3. **API inconsistency**: The property getter returns `float`, but the setter only accepts `int` (plus `math.inf`)
4. **Documentation mismatch**: The docstring doesn't mention this restriction, only stating "The total number of tokens available for borrowing"

The fact that `math.inf` (a float) is accepted shows that float handling is partially implemented, making the rejection of other float values even more inconsistent.

## Relevant Context

- The `CapacityLimiter` is used for rate limiting and resource management in async operations
- The adapter pattern is used to provide a synchronous interface when no async backend is available
- Version 3.0 made `total_tokens` writable (per docstring at line 520-521)
- The validation also checks that `value >= 1` (line 647-648), which would work fine with floats

Code location: `/home/npc/miniconda/lib/python3.13/site-packages/anyio/_core/_synchronization.py`

Documentation reference: The property docstring (lines 513-523) doesn't mention the integer-only restriction.

## Proposed Fix

The most straightforward fix is to accept float values as the type hints promise:

```diff
--- a/anyio/_core/_synchronization.py
+++ b/anyio/_core/_synchronization.py
@@ -642,8 +642,10 @@ class CapacityLimiterAdapter(CapacityLimiter):

     @total_tokens.setter
     def total_tokens(self, value: float) -> None:
-        if not isinstance(value, int) and value is not math.inf:
-            raise TypeError("total_tokens must be an int or math.inf")
+        if not isinstance(value, (int, float)):
+            raise TypeError("total_tokens must be a number")
+        elif math.isnan(value):
+            raise ValueError("total_tokens cannot be NaN")
         elif value < 1:
             raise ValueError("total_tokens must be >= 1")
```

This change would:
1. Accept all numeric values (int and float) as the type hints promise
2. Add explicit NaN checking for robustness
3. Maintain backward compatibility (all previously accepted values still work)
4. Make the implementation consistent with the type annotations