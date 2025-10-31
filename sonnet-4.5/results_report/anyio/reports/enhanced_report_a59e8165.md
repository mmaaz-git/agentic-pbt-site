# Bug Report: anyio.CapacityLimiter Rejects Valid Infinity Representations

**Target**: `anyio.CapacityLimiter` (specifically `CapacityLimiterAdapter.total_tokens` setter)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`CapacityLimiter` uses identity comparison (`is`) instead of equality comparison (`==`) when checking for infinity, causing it to reject `float('inf')` and `float('infinity')` even though they are mathematically equal to `math.inf`.

## Property-Based Test

```python
from anyio import CapacityLimiter
from hypothesis import given, strategies as st
import math

@given(st.sampled_from([math.inf, float('inf'), float('infinity')]))
def test_capacity_limiter_accepts_all_infinity_representations(inf_value):
    limiter = CapacityLimiter(inf_value)
    assert limiter.total_tokens == inf_value

if __name__ == "__main__":
    test_capacity_limiter_accepts_all_infinity_representations()
```

<details>

<summary>
**Failing input**: `inf` (which is `float('inf')`)
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
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 11, in <module>
    test_capacity_limiter_accepts_all_infinity_representations()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 6, in test_capacity_limiter_accepts_all_infinity_representations
    def test_capacity_limiter_accepts_all_infinity_representations(inf_value):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 7, in test_capacity_limiter_accepts_all_infinity_representations
    limiter = CapacityLimiter(inf_value)
  File "/home/npc/miniconda/lib/python3.13/site-packages/anyio/_core/_synchronization.py", line 498, in __new__
    return CapacityLimiterAdapter(total_tokens)
  File "/home/npc/miniconda/lib/python3.13/site-packages/anyio/_core/_synchronization.py", line 614, in __init__
    self.total_tokens = total_tokens
    ^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/anyio/_core/_synchronization.py", line 646, in total_tokens
    raise TypeError("total_tokens must be an int or math.inf")
TypeError: total_tokens must be an int or math.inf
Falsifying example: test_capacity_limiter_accepts_all_infinity_representations(
    inf_value=inf,
)
```
</details>

## Reproducing the Bug

```python
import math
from anyio import CapacityLimiter

# Show that float('inf') and math.inf are equal but not identical
print(f"float('inf') == math.inf: {float('inf') == math.inf}")
print(f"float('inf') is math.inf: {float('inf') is math.inf}")
print()

# Try to create CapacityLimiter with float('inf')
print("Creating CapacityLimiter with float('inf'):")
try:
    limiter = CapacityLimiter(float('inf'))
    print("Success - limiter created")
except TypeError as e:
    print(f"Error: {e}")
print()

# Try to create CapacityLimiter with math.inf
print("Creating CapacityLimiter with math.inf:")
try:
    limiter = CapacityLimiter(math.inf)
    print("Success - limiter created")
except TypeError as e:
    print(f"Error: {e}")
```

<details>

<summary>
TypeError when using float('inf') but success with math.inf
</summary>
```
float('inf') == math.inf: True
float('inf') is math.inf: False

Creating CapacityLimiter with float('inf'):
Error: total_tokens must be an int or math.inf

Creating CapacityLimiter with math.inf:
Success - limiter created
```
</details>

## Why This Is A Bug

This violates expected Python behavior in several ways:

1. **Mathematical Equivalence**: In Python, `float('inf') == math.inf` returns True. They represent the same mathematical value (positive infinity) and should be treated equivalently.

2. **Backend Inconsistency**: The asyncio backend (at line 1963 in `_backends/_asyncio.py`) uses `math.isinf(value)` which correctly accepts any infinity representation, while `CapacityLimiterAdapter` (at line 645 in `_core/_synchronization.py`) uses identity comparison `value is not math.inf`.

3. **Type System Contradiction**: The function signature accepts `float` type, and `float('inf')` is a valid float. The type system suggests any float infinity should work.

4. **Common Python Patterns**: Users may obtain infinity from various legitimate sources:
   - Parsing JSON/YAML: `float('inf')`
   - Mathematical operations: `float('inf') / 2` still gives infinity
   - External libraries that return `float('inf')`

5. **Documentation Ambiguity**: The error message says "must be an int or math.inf" which doesn't explicitly require the exact `math.inf` object - a reasonable interpretation is that any infinity value should work.

## Relevant Context

The bug is located in `/anyio/_core/_synchronization.py` at line 645:
```python
if not isinstance(value, int) and value is not math.inf:
    raise TypeError("total_tokens must be an int or math.inf")
```

The asyncio backend shows the correct implementation at `/anyio/_backends/_asyncio.py` line 1963:
```python
if not isinstance(value, int) and not math.isinf(value):
    raise TypeError("total_tokens must be an int or math.inf")
```

This inconsistency suggests the identity check in `CapacityLimiterAdapter` is likely unintentional. The anyio documentation does not explicitly require the exact `math.inf` object, and the type annotations accept `float` which would include any float infinity.

Documentation: https://anyio.readthedocs.io/en/stable/api.html#anyio.CapacityLimiter

## Proposed Fix

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