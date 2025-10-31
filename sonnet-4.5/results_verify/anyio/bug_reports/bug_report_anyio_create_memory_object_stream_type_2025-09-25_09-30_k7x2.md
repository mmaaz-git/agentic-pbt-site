# Bug Report: anyio.create_memory_object_stream Type Annotation Mismatch

**Target**: `anyio.create_memory_object_stream`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `create_memory_object_stream` function has a type annotation that claims `max_buffer_size: float`, but the implementation only accepts integers or `math.inf`, rejecting all other float values. This creates a contract violation between the type signature and runtime behavior.

## Property-Based Test

```python
from hypothesis import given, strategies as st

@given(st.floats(min_value=0.0, max_value=1000.0))
def test_buffer_size_accepts_float(buffer_size):
    async def check():
        send_stream, receive_stream = anyio.create_memory_object_stream(max_buffer_size=buffer_size)
        assert send_stream.statistics().max_buffer_size == buffer_size

    anyio.run(check)
```

**Failing input**: `buffer_size=0.5` (or any non-integer float)

## Reproducing the Bug

```python
import anyio

send, recv = anyio.create_memory_object_stream(max_buffer_size=10.5)
```

Output:
```
ValueError: max_buffer_size must be either an integer or math.inf
```

## Why This Is A Bug

The function signature declares `max_buffer_size: float`, which signals to type checkers and users that any float value is acceptable. However, the implementation at `anyio/_core/_streams.py:38-40` validates:

```python
if max_buffer_size != math.inf and not isinstance(max_buffer_size, int):
    raise ValueError("max_buffer_size must be either an integer or math.inf")
```

This creates a contract violation where:
1. Type checkers (mypy, pyright) will allow passing any float
2. Runtime will reject non-integer floats (except `math.inf`)
3. Users following the type hints will encounter unexpected ValueErrors

## Fix

```diff
--- a/anyio/_core/_streams.py
+++ b/anyio/_core/_streams.py
@@ -1,6 +1,7 @@
 from __future__ import annotations

 import math
+from typing import Union
 from typing import TypeVar
 from warnings import warn

@@ -35,7 +36,7 @@ class create_memory_object_stream(
     """

     def __new__(  # type: ignore[misc]
-        cls, max_buffer_size: float = 0, item_type: object = None
+        cls, max_buffer_size: int | float = 0, item_type: object = None
     ) -> tuple[MemoryObjectSendStream[T_Item], MemoryObjectReceiveStream[T_Item]]:
         if max_buffer_size != math.inf and not isinstance(max_buffer_size, int):
             raise ValueError("max_buffer_size must be either an integer or math.inf")
```

Note: An even better fix would be to use a more precise type annotation like `int | Literal[math.inf]` (though this requires typing_extensions for proper support), or create a custom type alias that better expresses the constraint.