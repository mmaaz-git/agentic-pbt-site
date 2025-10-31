# Bug Report: anyio.CapacityLimiter total_tokens Type Contract Violation

**Target**: `anyio.CapacityLimiter`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `CapacityLimiter` class declares `total_tokens` parameter with type `float`, but the runtime validation rejects non-integer float values, causing a `TypeError` for valid inputs according to the type annotation.

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

**Failing input**: `2.5`

## Reproducing the Bug

```python
import anyio

limiter = anyio.CapacityLimiter(2.5)
```

Expected: Creates a limiter with 2.5 tokens (as type annotation declares `total_tokens: float`)
Actual: Raises `TypeError: total_tokens must be an int or math.inf`

Also fails when setting the property:
```python
import anyio

limiter = anyio.CapacityLimiter(1)
limiter.total_tokens = 3.7
```

Raises: `TypeError: total_tokens must be an int or math.inf`

## Why This Is A Bug

The type annotation on both `__new__` and the `total_tokens` setter declare the parameter as `float`:

```python
def __new__(cls, total_tokens: float) -> CapacityLimiter:
    ...

@total_tokens.setter
def total_tokens(self, value: float) -> None:
    ...
```

However, the runtime validation in `CapacityLimiterAdapter.total_tokens` setter (line 645-646) only accepts `int` or `math.inf`:

```python
if not isinstance(value, int) and value is not math.inf:
    raise TypeError("total_tokens must be an int or math.inf")
```

This violates the API contract established by the type annotations. Users relying on type hints will pass valid float values and encounter unexpected TypeErrors.

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

Alternatively, if the intention is to only accept integers and infinity, the type annotation should be changed to reflect this:

```diff
--- a/anyio/_core/_synchronization.py
+++ b/anyio/_core/_synchronization.py
@@ -493,7 +493,7 @@ class SemaphoreAdapter(Semaphore):

 class CapacityLimiter:
-    def __new__(cls, total_tokens: float) -> CapacityLimiter:
+    def __new__(cls, total_tokens: int | float) -> CapacityLimiter:
         try:
             return get_async_backend().create_capacity_limiter(total_tokens)
         except AsyncLibraryNotFoundError:
@@ -511,7 +511,7 @@ class CapacityLimiter:
     @property
     def total_tokens(self) -> float:
         """
-        The total number of tokens available for borrowing.
+        The total number of tokens available for borrowing (must be int or math.inf).

         This is a read-write property. If the total number of tokens is increased, the
         proportionate number of tasks waiting on this limiter will be granted their
@@ -636,7 +636,7 @@ class CapacityLimiterAdapter(CapacityLimiter):
     @property
-    def total_tokens(self) -> float:
+    def total_tokens(self) -> int | float:
         if self._internal_limiter is None:
             return self._total_tokens

@@ -642,7 +642,7 @@ class CapacityLimiterAdapter(CapacityLimiter):

     @total_tokens.setter
-    def total_tokens(self, value: float) -> None:
+    def total_tokens(self, value: int | float) -> None:
         if not isinstance(value, int) and value is not math.inf:
             raise TypeError("total_tokens must be an int or math.inf")
         elif value < 1:
```