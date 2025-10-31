# Bug Report: anyio.create_memory_object_stream rejects float despite type hint

**Target**: `anyio.create_memory_object_stream`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`create_memory_object_stream` has type hint `max_buffer_size: float` but raises ValueError when given float values like `1.0`, only accepting `int` or `math.inf`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pytest
from anyio import create_memory_object_stream

@given(max_buffer_size=st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False))
def test_create_memory_object_stream_accepts_float(max_buffer_size):
    send, recv = create_memory_object_stream(max_buffer_size)
    assert send.statistics().max_buffer_size == max_buffer_size
    send.close()
    recv.close()
```

**Failing input**: `max_buffer_size=1.0`

## Reproducing the Bug

```python
import math
from anyio import create_memory_object_stream

create_memory_object_stream(1)
create_memory_object_stream(math.inf)

create_memory_object_stream(1.0)
```

Output:
```
ValueError: max_buffer_size must be either an integer or math.inf
```

## Why This Is A Bug

The function signature declares `max_buffer_size: float = 0` (line 37 in `anyio/_core/_streams.py`), indicating it should accept any float value. However, the implementation (line 39) rejects all floats except `math.inf`:

```python
if max_buffer_size != math.inf and not isinstance(max_buffer_size, int):
    raise ValueError("max_buffer_size must be either an integer or math.inf")
```

This violates the type contract. Either:
1. The type hint should be `int | float` where float means only `math.inf`, or
2. The implementation should accept arbitrary non-negative floats

## Fix

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

Alternatively, accept floats and truncate to int:

```diff
--- a/anyio/_core/_streams.py
+++ b/anyio/_core/_streams.py
@@ -36,7 +36,9 @@ class create_memory_object_stream(
     def __new__(  # type: ignore[misc]
         cls, max_buffer_size: float = 0, item_type: object = None
     ) -> tuple[MemoryObjectSendStream[T_Item], MemoryObjectReceiveStream[T_Item]]:
-        if max_buffer_size != math.inf and not isinstance(max_buffer_size, int):
+        if max_buffer_size != math.inf:
+            max_buffer_size = int(max_buffer_size)
+        if not isinstance(max_buffer_size, (int, float)):
             raise ValueError("max_buffer_size must be either an integer or math.inf")
         if max_buffer_size < 0:
             raise ValueError("max_buffer_size cannot be negative")
```