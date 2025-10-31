# Bug Report: anyio.create_memory_object_stream Float Type Rejection

**Target**: `anyio.create_memory_object_stream`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The function signature declares `max_buffer_size: float` but the implementation rejects all float values except `math.inf`, violating the type contract.

## Property-Based Test

```python
from anyio import create_memory_object_stream
from hypothesis import given, strategies as st


@given(max_buffer=st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False))
def test_create_memory_object_stream_accepts_floats(max_buffer):
    send, receive = create_memory_object_stream(max_buffer_size=max_buffer)
```

**Failing input**: `max_buffer=1.0`

## Reproducing the Bug

```python
from anyio import create_memory_object_stream

send, receive = create_memory_object_stream(max_buffer_size=1.0)
```

Output:
```
ValueError: max_buffer_size must be either an integer or math.inf
```

## Why This Is A Bug

The function signature explicitly declares the parameter type as `float`:
```python
def __new__(cls, max_buffer_size: float = 0, item_type: object = None):
```

However, the implementation at `anyio/_core/_streams.py:40` rejects all floats except `math.inf`:
```python
if max_buffer_size != math.inf and not isinstance(max_buffer_size, int):
    raise ValueError("max_buffer_size must be either an integer or math.inf")
```

This violates the type contract. Users relying on type hints would expect any float to be accepted, but values like `1.0`, `5.5`, or `0.0` are rejected even though they satisfy the type signature.

## Fix

Either:
1. Change the type annotation to `int | float` and accept floats by converting them (e.g., via `int(max_buffer_size)`), or
2. Change the type annotation to `int` to match the actual implementation

Option 2 is simpler and matches the current behavior:

```diff
--- a/anyio/_core/_streams.py
+++ b/anyio/_core/_streams.py
@@ -37,7 +37,7 @@ class create_memory_object_stream(
     """

     def __new__(  # type: ignore[misc]
-        cls, max_buffer_size: float = 0, item_type: object = None
+        cls, max_buffer_size: int | float = 0, item_type: object = None
     ) -> tuple[MemoryObjectSendStream[T_Item], MemoryObjectReceiveStream[T_Item]]:
         if max_buffer_size != math.inf and not isinstance(max_buffer_size, int):
```

Alternatively, accept floats:
```diff
--- a/anyio/_core/_streams.py
+++ b/anyio/_core/_streams.py
@@ -37,8 +37,11 @@ class create_memory_object_stream(
     """

     def __new__(  # type: ignore[misc]
         cls, max_buffer_size: float = 0, item_type: object = None
     ) -> tuple[MemoryObjectSendStream[T_Item], MemoryObjectReceiveStream[T_Item]]:
+        if isinstance(max_buffer_size, float) and max_buffer_size != math.inf:
+            max_buffer_size = int(max_buffer_size)
+
         if max_buffer_size != math.inf and not isinstance(max_buffer_size, int):
             raise ValueError("max_buffer_size must be either an integer or math.inf")
```