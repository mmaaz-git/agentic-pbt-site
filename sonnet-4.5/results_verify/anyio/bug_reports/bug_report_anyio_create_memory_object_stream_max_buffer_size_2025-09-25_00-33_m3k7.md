# Bug Report: anyio.create_memory_object_stream max_buffer_size Type Contract Violation

**Target**: `anyio.create_memory_object_stream`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `create_memory_object_stream` function declares `max_buffer_size` parameter with type `float`, but the runtime validation rejects non-integer float values, causing a `ValueError` for valid inputs according to the type annotation.

## Property-Based Test

```python
import math
from hypothesis import given, strategies as st
import anyio


@given(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False).filter(lambda x: x > 0 and not x.is_integer()))
def test_memory_stream_accepts_float_buffer_size(max_buffer_size):
    send, recv = anyio.create_memory_object_stream(max_buffer_size)
```

**Failing input**: `1.5`

## Reproducing the Bug

```python
import anyio

send, recv = anyio.create_memory_object_stream(2.5)
```

Expected: Creates a memory object stream with buffer size 2.5 (as type annotation declares `max_buffer_size: float`)
Actual: Raises `ValueError: max_buffer_size must be either an integer or math.inf`

## Why This Is A Bug

The type annotation declares the parameter as `float`:

```python
def __new__(
    cls, max_buffer_size: float = 0, item_type: object = None
) -> tuple[MemoryObjectSendStream[T_Item], MemoryObjectReceiveStream[T_Item]]:
```

However, the runtime validation (line 39-40) only accepts `int` or `math.inf`:

```python
if max_buffer_size != math.inf and not isinstance(max_buffer_size, int):
    raise ValueError("max_buffer_size must be either an integer or math.inf")
```

This violates the API contract established by the type annotation. Users relying on type hints will pass valid float values (e.g., `2.5`) and encounter unexpected ValueErrors.

## Fix

The type annotation should be corrected to reflect the actual validation:

```diff
--- a/anyio/_core/_streams.py
+++ b/anyio/_core/_streams.py
@@ -34,7 +34,7 @@ class create_memory_object_stream(
     """

     def __new__(  # type: ignore[misc]
-        cls, max_buffer_size: float = 0, item_type: object = None
+        cls, max_buffer_size: int | float = 0, item_type: object = None
     ) -> tuple[MemoryObjectSendStream[T_Item], MemoryObjectReceiveStream[T_Item]]:
         if max_buffer_size != math.inf and not isinstance(max_buffer_size, int):
             raise ValueError("max_buffer_size must be either an integer or math.inf")
```

Additionally, the docstring should clarify that only `int` or `math.inf` are accepted:

```diff
@@ -20,7 +20,8 @@ class create_memory_object_stream(
     """
     Create a memory object stream.

     The stream's item type can be annotated like
     :func:`create_memory_object_stream[T_Item]`.

-    :param max_buffer_size: number of items held in the buffer until ``send()`` starts
+    :param max_buffer_size: number of items held in the buffer (must be an integer or
+        math.inf) until ``send()`` starts
         blocking
```