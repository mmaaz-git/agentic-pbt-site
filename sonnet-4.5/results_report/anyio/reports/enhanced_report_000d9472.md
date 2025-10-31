# Bug Report: anyio.create_memory_object_stream Type Contract Violation for Float Parameters

**Target**: `anyio.create_memory_object_stream`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The function signature declares `max_buffer_size: float` but the implementation rejects all float values except `math.inf`, violating the type contract that developers rely on for IDE support and type checking.

## Property-Based Test

```python
from anyio import create_memory_object_stream
from hypothesis import given, strategies as st


@given(max_buffer=st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False))
def test_create_memory_object_stream_accepts_floats(max_buffer):
    send, receive = create_memory_object_stream(max_buffer_size=max_buffer)


# Run the test
test_create_memory_object_stream_accepts_floats()
```

<details>

<summary>
**Failing input**: `max_buffer=1.0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 11, in <module>
    test_create_memory_object_stream_accepts_floats()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 6, in test_create_memory_object_stream_accepts_floats
    def test_create_memory_object_stream_accepts_floats(max_buffer):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 7, in test_create_memory_object_stream_accepts_floats
    send, receive = create_memory_object_stream(max_buffer_size=max_buffer)
                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/anyio/_core/_streams.py", line 40, in __new__
    raise ValueError("max_buffer_size must be either an integer or math.inf")
ValueError: max_buffer_size must be either an integer or math.inf
Falsifying example: test_create_memory_object_stream_accepts_floats(
    max_buffer=1.0,
)
```
</details>

## Reproducing the Bug

```python
from anyio import create_memory_object_stream

# Test with float value 1.0
try:
    send, receive = create_memory_object_stream(max_buffer_size=1.0)
    print("Successfully created stream with max_buffer_size=1.0")
except ValueError as e:
    print(f"Error with max_buffer_size=1.0: {e}")

# Test with float value 5.5
try:
    send, receive = create_memory_object_stream(max_buffer_size=5.5)
    print("Successfully created stream with max_buffer_size=5.5")
except ValueError as e:
    print(f"Error with max_buffer_size=5.5: {e}")

# Test with float value 0.0
try:
    send, receive = create_memory_object_stream(max_buffer_size=0.0)
    print("Successfully created stream with max_buffer_size=0.0")
except ValueError as e:
    print(f"Error with max_buffer_size=0.0: {e}")

# Test with int value 1 (should work)
try:
    send, receive = create_memory_object_stream(max_buffer_size=1)
    print("Successfully created stream with max_buffer_size=1 (int)")
except ValueError as e:
    print(f"Error with max_buffer_size=1 (int): {e}")

# Test with math.inf (should work)
import math
try:
    send, receive = create_memory_object_stream(max_buffer_size=math.inf)
    print("Successfully created stream with max_buffer_size=math.inf")
except ValueError as e:
    print(f"Error with max_buffer_size=math.inf: {e}")
```

<details>

<summary>
ValueError raised for all float values except math.inf
</summary>
```
Error with max_buffer_size=1.0: max_buffer_size must be either an integer or math.inf
Error with max_buffer_size=5.5: max_buffer_size must be either an integer or math.inf
Error with max_buffer_size=0.0: max_buffer_size must be either an integer or math.inf
Successfully created stream with max_buffer_size=1 (int)
Successfully created stream with max_buffer_size=math.inf
```
</details>

## Why This Is A Bug

This violates the explicit type contract defined in the function signature. At line 37 of `/home/npc/pbt/agentic-pbt/envs/anyio_env/lib/python3.13/site-packages/anyio/_core/_streams.py`, the function declares:

```python
def __new__(cls, max_buffer_size: float = 0, item_type: object = None)
```

The `float` type annotation creates a contract that promises the function will accept float values. However, the implementation at lines 39-40 explicitly rejects all float values except `math.inf`:

```python
if max_buffer_size != math.inf and not isinstance(max_buffer_size, int):
    raise ValueError("max_buffer_size must be either an integer or math.inf")
```

This breaks several Python conventions and developer expectations:
1. **Type checker compatibility**: Tools like mypy would validate `create_memory_object_stream(max_buffer_size=1.0)` as correct based on the type hint
2. **IDE autocomplete**: IDEs show `float` as the expected type, misleading developers
3. **API consistency**: The function accepts one specific float (`math.inf`) but rejects all others, creating an inconsistent interface
4. **Python typing standards**: In Python's type system, declaring a parameter as `float` means it should accept any float value

The docstring at line 25 describes the parameter as "number of items held in the buffer" but doesn't specify type restrictions, leaving the type annotation as the primary source of truth for developers.

## Relevant Context

This bug affects the core stream creation functionality in AnyIO, a popular asynchronous I/O library used in many Python projects. The memory object streams are commonly used for inter-task communication in async applications.

The inconsistency is particularly problematic because:
- The default value `0` is actually an integer, not a float
- The special case acceptance of `math.inf` shows the function can handle floats
- Buffer sizes logically represent item counts (integers), but the type signature promises float support

Related documentation:
- AnyIO streams documentation: https://anyio.readthedocs.io/en/stable/streams.html
- Source code: https://github.com/agronholm/anyio/blob/master/src/anyio/_core/_streams.py

## Proposed Fix

The cleanest fix is to update the type annotation to match the actual implementation:

```diff
--- a/anyio/_core/_streams.py
+++ b/anyio/_core/_streams.py
@@ -3,6 +3,7 @@ from __future__ import annotations

 import math
 from typing import TypeVar
+from typing import Union
 from warnings import warn

 from ..streams.memory import (
@@ -34,7 +35,7 @@ class create_memory_object_stream(
     """

     def __new__(  # type: ignore[misc]
-        cls, max_buffer_size: float = 0, item_type: object = None
+        cls, max_buffer_size: Union[int, float] = 0, item_type: object = None
     ) -> tuple[MemoryObjectSendStream[T_Item], MemoryObjectReceiveStream[T_Item]]:
         if max_buffer_size != math.inf and not isinstance(max_buffer_size, int):
             raise ValueError("max_buffer_size must be either an integer or math.inf")
```

Alternatively, if maintaining backward compatibility with the float type hint is important, accept and convert float values:

```diff
--- a/anyio/_core/_streams.py
+++ b/anyio/_core/_streams.py
@@ -36,6 +36,9 @@ class create_memory_object_stream(
     def __new__(  # type: ignore[misc]
         cls, max_buffer_size: float = 0, item_type: object = None
     ) -> tuple[MemoryObjectSendStream[T_Item], MemoryObjectReceiveStream[T_Item]]:
+        # Convert float to int if it's not math.inf
+        if isinstance(max_buffer_size, float) and max_buffer_size != math.inf:
+            max_buffer_size = int(max_buffer_size)
         if max_buffer_size != math.inf and not isinstance(max_buffer_size, int):
             raise ValueError("max_buffer_size must be either an integer or math.inf")
         if max_buffer_size < 0:
```