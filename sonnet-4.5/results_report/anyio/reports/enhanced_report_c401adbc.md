# Bug Report: anyio.CapacityLimiter total_tokens Type Contract Violation

**Target**: `anyio.CapacityLimiter`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `CapacityLimiter` class type annotations declare `total_tokens` parameter as `float`, but the runtime validation rejects non-integer float values with a `TypeError`, violating the type contract promised by the API.

## Property-Based Test

```python
import math
from hypothesis import given, strategies as st
import anyio


@given(st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False).filter(lambda x: not x.is_integer()))
def test_capacity_limiter_accepts_float_tokens(total_tokens):
    limiter = anyio.CapacityLimiter(total_tokens)
    assert limiter.total_tokens == total_tokens
```

<details>

<summary>
**Failing input**: `1.5`
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
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 13, in <module>
    test_capacity_limiter_accepts_float_tokens()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 7, in test_capacity_limiter_accepts_float_tokens
    def test_capacity_limiter_accepts_float_tokens(total_tokens):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 8, in test_capacity_limiter_accepts_float_tokens
    limiter = anyio.CapacityLimiter(total_tokens)
  File "/home/npc/miniconda/lib/python3.13/site-packages/anyio/_core/_synchronization.py", line 498, in __new__
    return CapacityLimiterAdapter(total_tokens)
  File "/home/npc/miniconda/lib/python3.13/site-packages/anyio/_core/_synchronization.py", line 614, in __init__
    self.total_tokens = total_tokens
    ^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/anyio/_core/_synchronization.py", line 646, in total_tokens
    raise TypeError("total_tokens must be an int or math.inf")
TypeError: total_tokens must be an int or math.inf
Falsifying example: test_capacity_limiter_accepts_float_tokens(
    total_tokens=1.5,
)
```
</details>

## Reproducing the Bug

```python
import anyio

# Test 1: Creating a CapacityLimiter with a non-integer float value
print("Test 1: Creating CapacityLimiter with float 2.5")
try:
    limiter = anyio.CapacityLimiter(2.5)
    print(f"SUCCESS: Created limiter with {limiter.total_tokens} tokens")
except TypeError as e:
    print(f"FAILED: {e}")

print("\n" + "="*50 + "\n")

# Test 2: Setting total_tokens property to a non-integer float value
print("Test 2: Setting total_tokens property to float 3.7")
try:
    limiter = anyio.CapacityLimiter(1)
    print(f"Created limiter with {limiter.total_tokens} tokens")
    limiter.total_tokens = 3.7
    print(f"SUCCESS: Updated limiter to {limiter.total_tokens} tokens")
except TypeError as e:
    print(f"FAILED when setting property: {e}")

print("\n" + "="*50 + "\n")

# Test 3: Creating with integer (should work)
print("Test 3: Creating CapacityLimiter with integer 5")
try:
    limiter = anyio.CapacityLimiter(5)
    print(f"SUCCESS: Created limiter with {limiter.total_tokens} tokens")
except Exception as e:
    print(f"FAILED: {e}")

print("\n" + "="*50 + "\n")

# Test 4: Creating with math.inf (should work)
print("Test 4: Creating CapacityLimiter with math.inf")
import math
try:
    limiter = anyio.CapacityLimiter(math.inf)
    print(f"SUCCESS: Created limiter with {limiter.total_tokens} tokens")
except Exception as e:
    print(f"FAILED: {e}")
```

<details>

<summary>
TypeError raised when creating CapacityLimiter with float 2.5 and setting property to 3.7
</summary>
```
Test 1: Creating CapacityLimiter with float 2.5
FAILED: total_tokens must be an int or math.inf

==================================================

Test 2: Setting total_tokens property to float 3.7
Created limiter with 1 tokens
FAILED when setting property: total_tokens must be an int or math.inf

==================================================

Test 3: Creating CapacityLimiter with integer 5
SUCCESS: Created limiter with 5 tokens

==================================================

Test 4: Creating CapacityLimiter with math.inf
SUCCESS: Created limiter with inf tokens
```
</details>

## Why This Is A Bug

This violates the explicit type contract established by anyio's public API. The type annotations clearly declare that `total_tokens` accepts a `float` parameter in three key locations:

1. **`CapacityLimiter.__new__` method** (line 494 in `/anyio/_core/_synchronization.py`):
   ```python
   def __new__(cls, total_tokens: float) -> CapacityLimiter:
   ```

2. **`CapacityLimiterAdapter.__init__` method** (line 613):
   ```python
   def __init__(self, total_tokens: float) -> None:
   ```

3. **`total_tokens` property setter** (line 644):
   ```python
   @total_tokens.setter
   def total_tokens(self, value: float) -> None:
   ```

However, the runtime validation in the `CapacityLimiterAdapter.total_tokens` setter (lines 645-646) contradicts these annotations:
```python
if not isinstance(value, int) and value is not math.inf:
    raise TypeError("total_tokens must be an int or math.inf")
```

This creates a situation where:
- Developers using type checking tools (mypy, pyright, etc.) will see that `float` values are acceptable
- The documentation and docstrings do not mention this restriction
- Valid float values like `2.5` that pass type checking will fail at runtime
- Inconsistently, `math.inf` (which is a float) is accepted, but other float values are not

The bug represents a breach of the Liskov Substitution Principle - code that is type-correct according to the declared interface fails at runtime.

## Relevant Context

- **Affected versions**: Current version of anyio (verified with the latest available)
- **Python version tested**: Python 3.13
- **Location of validation**: `/anyio/_core/_synchronization.py:645-646`
- **Conceptual validity**: Fractional capacity limits are meaningful in real-world scenarios (e.g., rate limiting to 2.5 operations per second)
- **Documentation gap**: Neither the API reference nor the synchronization guide mentions this integer-only constraint
- **Workaround available**: Users can work around this by using only integer values or `math.inf`

The issue appears when anyio falls back to `CapacityLimiterAdapter` (when not in an async context), which is where the restrictive validation occurs. The backend implementations may have different behavior.

## Proposed Fix

The most straightforward fix is to accept all float values in the validation, while still rejecting NaN:

```diff
--- a/anyio/_core/_synchronization.py
+++ b/anyio/_core/_synchronization.py
@@ -642,8 +642,8 @@ class CapacityLimiterAdapter(CapacityLimiter):

     @total_tokens.setter
     def total_tokens(self, value: float) -> None:
-        if not isinstance(value, int) and value is not math.inf:
-            raise TypeError("total_tokens must be an int or math.inf")
+        if not isinstance(value, (int, float)) or (isinstance(value, float) and math.isnan(value)):
+            raise TypeError("total_tokens must be a number (int or float, not NaN)")
         elif value < 1:
             raise ValueError("total_tokens must be >= 1")
```