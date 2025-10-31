# Bug Report: anyio.create_memory_object_stream Type Annotation Mismatch

**Target**: `anyio.create_memory_object_stream`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `max_buffer_size` parameter has type annotation `float` but runtime validation only accepts `int` or `math.inf`, creating a contract violation between static type checking and runtime behavior.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import math
from anyio import create_memory_object_stream

@given(st.floats(min_value=0, max_value=1000).filter(
    lambda x: x != math.inf and not (isinstance(x, float) and x == int(x)) and not math.isnan(x)
))
def test_max_buffer_size_type_contract(value):
    """
    Test that max_buffer_size accepts all non-negative floats as per type annotation.
    The type annotation says 'float', so all floats should be valid.
    """
    send_stream, receive_stream = create_memory_object_stream(value)
    send_stream.close()
    receive_stream.close()
```

**Failing input**: Any float that is not an integer or `math.inf`, e.g., `5.5`, `3.14`, `0.5`

## Reproducing the Bug

```python
from anyio import create_memory_object_stream

try:
    send, recv = create_memory_object_stream(5.5)
    print("Accepted 5.5")
except ValueError as e:
    print(f"Rejected 5.5: {e}")

try:
    send, recv = create_memory_object_stream(5)
    print("Accepted 5 (int)")
except ValueError as e:
    print(f"Rejected 5: {e}")
```

Output:
```
Rejected 5.5: max_buffer_size must be either an integer or math.inf
Accepted 5 (int)
```

## Why This Is A Bug

In `/lib/python3.13/site-packages/anyio/_core/_streams.py`, the function signature at line 36-38 is:

```python
def __new__(
    cls, max_buffer_size: float = 0, item_type: object = None
) -> tuple[MemoryObjectSendStream[T_Item], MemoryObjectReceiveStream[T_Item]]:
```

The type annotation `max_buffer_size: float` tells static type checkers that **any float value** is acceptable. Type checkers like mypy and pyright will not warn if you pass `5.5` to this function.

However, the runtime validation at lines 39-40 contradicts this:

```python
if max_buffer_size != math.inf and not isinstance(max_buffer_size, int):
    raise ValueError("max_buffer_size must be either an integer or math.inf")
```

This validation **rejects** float values like `5.5`, even though the type annotation claims they are valid. This creates a contract violation where code that passes static type checking will fail at runtime.

**Impact**: Users relying on type annotations will write code like:
```python
buffer_size: float = get_buffer_size()  # returns 5.5
create_memory_object_stream(buffer_size)  # Type checks pass, but runtime fails!
```

## Fix

Option 1: Make type annotation match validation (recommended)

```diff
 def __new__(
-    cls, max_buffer_size: float = 0, item_type: object = None
+    cls, max_buffer_size: int | float = 0, item_type: object = None
 ) -> tuple[MemoryObjectSendStream[T_Item], MemoryObjectReceiveStream[T_Item]]:
+    """
+    :param max_buffer_size: number of items held in the buffer until send() starts
+        blocking. Must be a non-negative integer or math.inf.
+    """
     if max_buffer_size != math.inf and not isinstance(max_buffer_size, int):
         raise ValueError("max_buffer_size must be either an integer or math.inf")
```

Option 2: Make validation match type annotation

```diff
 def __new__(
     cls, max_buffer_size: float = 0, item_type: object = None
 ) -> tuple[MemoryObjectSendStream[T_Item], MemoryObjectReceiveStream[T_Item]]:
-    if max_buffer_size != math.inf and not isinstance(max_buffer_size, int):
-        raise ValueError("max_buffer_size must be either an integer or math.inf")
+    if not isinstance(max_buffer_size, (int, float)):
+        raise TypeError("max_buffer_size must be a number")
     if max_buffer_size < 0:
         raise ValueError("max_buffer_size cannot be negative")
+    if max_buffer_size != math.inf and max_buffer_size != int(max_buffer_size):
+        raise ValueError("max_buffer_size must be an integer or math.inf")
```

Option 1 is recommended as it documents the actual contract without changing behavior.