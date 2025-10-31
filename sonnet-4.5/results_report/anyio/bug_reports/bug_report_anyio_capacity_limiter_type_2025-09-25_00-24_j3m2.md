# Bug Report: anyio.CapacityLimiter Type Annotation Mismatch

**Target**: `anyio.CapacityLimiter` (specifically `CapacityLimiterAdapter`)
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `CapacityLimiter` constructor and `total_tokens` setter have type annotation `float` but runtime validation only accepts `int` or `math.inf`, creating a contract violation between static type checking and runtime behavior.

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
```

**Failing input**: Any float >= 1 that is not an integer or `math.inf`, e.g., `5.5`, `10.25`, `2.5`

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
```

Output:
```
Test 1: integer value
✓ Accepted 5 (int): total_tokens=5

Test 2: math.inf
✓ Accepted math.inf: total_tokens=inf

Test 3: float 5.5
✗ Rejected 5.5: total_tokens must be an int or math.inf
```

## Why This Is A Bug

In `/lib/python3.13/site-packages/anyio/_core/_synchronization.py`:

1. The `CapacityLimiterAdapter.__init__` signature at line 613:
```python
def __init__(self, total_tokens: float) -> None:
```

2. The `CapacityLimiter.__new__` signature at line 494:
```python
def __new__(cls, total_tokens: float) -> CapacityLimiter:
```

Both use type annotation `total_tokens: float`, telling static type checkers that **any float value** is acceptable.

However, the `total_tokens` setter at lines 643-649 contradicts this:

```python
@total_tokens.setter
def total_tokens(self, value: float) -> None:
    if not isinstance(value, int) and value is not math.inf:
        raise TypeError("total_tokens must be an int or math.inf")
    elif value < 1:
        raise ValueError("total_tokens must be >= 1")
```

This validation **rejects** float values like `5.5`, even though the type annotation claims they are valid.

**Impact**: Users relying on type annotations will write code like:
```python
capacity: float = calculate_capacity()  # returns 5.5
limiter = CapacityLimiter(capacity)  # Type checks pass, but runtime fails!
```

**Additional Issue**: Line 645 uses `value is not math.inf` instead of `value != math.inf`. While `is` works for `math.inf` (since it's a singleton), this is fragile and relies on implementation details. The `!=` operator is more appropriate for value comparisons.

## Fix

Option 1: Make type annotation match validation (recommended)

```diff
 class CapacityLimiter:
-    def __new__(cls, total_tokens: float) -> CapacityLimiter:
+    def __new__(cls, total_tokens: int | float) -> CapacityLimiter:
+        """
+        Create a capacity limiter.
+
+        :param total_tokens: the total number of tokens available for borrowing
+            (must be an integer >= 1 or math.inf)
+        """
         try:
             return get_async_backend().create_capacity_limiter(total_tokens)
         except AsyncLibraryNotFoundError:
             return CapacityLimiterAdapter(total_tokens)

 class CapacityLimiterAdapter(CapacityLimiter):
-    def __init__(self, total_tokens: float) -> None:
+    def __init__(self, total_tokens: int | float) -> None:
         self.total_tokens = total_tokens

     @total_tokens.setter
     def total_tokens(self, value: float) -> None:
-        if not isinstance(value, int) and value is not math.inf:
+        if not isinstance(value, int) and value != math.inf:
             raise TypeError("total_tokens must be an int or math.inf")
         elif value < 1:
             raise ValueError("total_tokens must be >= 1")
```

Option 2: Make validation match type annotation

```diff
 @total_tokens.setter
 def total_tokens(self, value: float) -> None:
-    if not isinstance(value, int) and value is not math.inf:
-        raise TypeError("total_tokens must be an int or math.inf")
+    if not isinstance(value, (int, float)):
+        raise TypeError("total_tokens must be a number")
     elif value < 1:
         raise ValueError("total_tokens must be >= 1")
+    elif value != math.inf and value != int(value):
+        raise ValueError("total_tokens must be an integer or math.inf")
```

Option 1 is recommended as it documents the actual contract without changing runtime behavior.