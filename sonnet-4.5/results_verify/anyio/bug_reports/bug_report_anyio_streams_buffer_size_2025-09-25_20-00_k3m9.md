# Bug Report: anyio.streams.memory Fractional Buffer Size Behavior

**Target**: `anyio.streams.memory.MemoryObjectStreamState`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

When `max_buffer_size` is set to a fractional value (e.g., 10.5), the actual buffer capacity is `ceil(max_buffer_size)` instead of the expected `floor(max_buffer_size)`, allowing more items than the documented "maximum number of items".

## Property-Based Test

```python
import anyio
import math
import pytest
from hypothesis import given, strategies as st, settings, assume
from anyio.streams.memory import (
    MemoryObjectSendStream,
    MemoryObjectReceiveStream,
    MemoryObjectStreamState,
)
from anyio import WouldBlock


@given(st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False))
@settings(max_examples=200)
def test_buffer_size_capacity_property(buffer_size):
    assume(buffer_size != math.floor(buffer_size))

    async def test_capacity():
        state = MemoryObjectStreamState[int](max_buffer_size=buffer_size)
        send_stream = MemoryObjectSendStream(state)
        receive_stream = MemoryObjectReceiveStream(state)

        expected_capacity = math.floor(buffer_size)

        for i in range(expected_capacity):
            send_stream.send_nowait(i)

        with pytest.raises(WouldBlock):
            send_stream.send_nowait(999)

        send_stream.close()
        receive_stream.close()

    anyio.run(test_capacity)
```

**Failing input**: `buffer_size=1.5` (or any fractional value)

## Reproducing the Bug

```python
import anyio
from anyio.streams.memory import (
    MemoryObjectSendStream,
    MemoryObjectReceiveStream,
    MemoryObjectStreamState,
)
from anyio import WouldBlock


async def main():
    state = MemoryObjectStreamState[int](max_buffer_size=10.5)
    send_stream = MemoryObjectSendStream(state)
    receive_stream = MemoryObjectReceiveStream(state)

    for i in range(11):
        send_stream.send_nowait(i)

    stats = send_stream.statistics()
    print(f"max_buffer_size: {stats.max_buffer_size}")
    print(f"current_buffer_used: {stats.current_buffer_used}")

    try:
        send_stream.send_nowait(11)
    except WouldBlock:
        print(f"Raised WouldBlock after {stats.current_buffer_used} items")
        print(
            f"Expected: floor(10.5) = 10 items\nActual: ceil(10.5) = 11 items"
        )

    send_stream.close()
    receive_stream.close()


anyio.run(main)
```

## Why This Is A Bug

The documentation for `create_memory_object_stream` states that `max_buffer_size` is the "number of items held in the buffer". A "number of items" should be an integer count. When a fractional value like 10.5 is provided:

- Expected behavior: Allow `floor(10.5) = 10` items (the maximum whole number of items ≤ 10.5)
- Actual behavior: Allow `ceil(10.5) = 11` items

This occurs because line 222 in `memory.py` uses the comparison `len(self._state.buffer) < self._state.max_buffer_size`. When `max_buffer_size=10.5` and the buffer has 10 items, the condition `10 < 10.5` is true, allowing an 11th item to be added.

The unintuitive behavior:
- buffer_size=10.1, 10.5, 10.9 all allow exactly 11 items
- buffer_size=0.1, 0.5, 0.9 all allow exactly 1 item

## Fix

Add validation to ensure `max_buffer_size` is either a positive integer or `math.inf`:

```diff
--- a/anyio/streams/memory.py
+++ b/anyio/streams/memory.py
@@ -1,6 +1,7 @@
 from __future__ import annotations

 import warnings
+import math
 from collections import OrderedDict, deque
 from dataclasses import dataclass, field
 from types import TracebackType
@@ -48,6 +49,11 @@ class MemoryObjectItemReceiver(Generic[T_Item]):
 @dataclass(eq=False)
 class MemoryObjectStreamState(Generic[T_Item]):
     max_buffer_size: float = field()
+
+    def __post_init__(self) -> None:
+        if not (math.isinf(self.max_buffer_size) or
+                (self.max_buffer_size >= 0 and self.max_buffer_size == int(self.max_buffer_size))):
+            raise ValueError("max_buffer_size must be a non-negative integer or math.inf")
+
     buffer: deque[T_Item] = field(init=False, default_factory=deque)
     open_send_channels: int = field(init=False, default=0)
     open_receive_channels: int = field(init=False, default=0)
```