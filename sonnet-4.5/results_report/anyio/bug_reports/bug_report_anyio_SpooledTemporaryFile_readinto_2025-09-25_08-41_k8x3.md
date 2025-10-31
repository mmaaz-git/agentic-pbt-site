# Bug Report: anyio SpooledTemporaryFile readinto Data Corruption

**Target**: `anyio.SpooledTemporaryFile.readinto` and `anyio.SpooledTemporaryFile.readinto1`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`SpooledTemporaryFile.readinto()` and `readinto1()` methods have two critical bugs that cause data corruption when the file hasn't been rolled to disk: (1) missing return statement causes double-read, (2) `readinto1()` calls wrong method.

## Property-Based Test

```python
import pytest
import anyio
from anyio import SpooledTemporaryFile


@pytest.mark.anyio
async def test_readinto_data_corruption():
    """
    Property: readinto() should read once and return bytes read.
    Currently it reads twice, corrupting the buffer.
    """
    async with SpooledTemporaryFile(max_size=1000, mode='w+b') as f:
        await f.write(b'Hello World!')
        await f.seek(0)

        buffer = bytearray(5)
        bytes_read = await f.readinto(buffer)

        # Expected: buffer contains first 5 bytes "Hello"
        # Actual: buffer contains bytes 6-10 " Worl" due to double-read
        assert buffer == bytearray(b' Worl')  # BUG confirmed
        assert buffer != bytearray(b'Hello')  # Should be this
```

**Failing input**: Any read from an in-memory SpooledTemporaryFile

## Reproducing the Bug

```python
import anyio
from anyio import SpooledTemporaryFile


async def demonstrate_bug():
    async with SpooledTemporaryFile(max_size=1000, mode='w+b') as f:
        await f.write(b'0123456789ABCDEF')
        await f.seek(0)

        buffer = bytearray(5)
        bytes_read = await f.readinto(buffer)

        print(f"Bytes read: {bytes_read}")
        print(f"Buffer contents: {buffer}")
        print(f"Expected: b'01234' (first 5 bytes)")
        print(f"Actual: {bytes(buffer)}")
        print(f"BUG: Data is offset! We got bytes 6-10 instead of 0-4")


anyio.run(demonstrate_bug)
```

Output:
```
Bytes read: 5
Buffer contents: bytearray(b'56789')
Expected: b'01234' (first 5 bytes)
Actual: b'56789'
BUG: Data is offset! We got bytes 6-10 instead of 0-4
```

## Why This Is A Bug

**Bug 1: Missing return in `readinto()`** (_tempfile.py lines 365-370):

```python
async def readinto(self: SpooledTemporaryFile[bytes], b: WriteableBuffer) -> int:
    if not self._rolled:
        await checkpoint_if_cancelled()
        self._fp.readinto(b)  # BUG: Missing 'return' statement!

    return await super().readinto(b)  # This executes ALWAYS, reading again!
```

When the file is in memory (`not self._rolled`), the code:
1. Calls `self._fp.readinto(b)` - fills buffer with first N bytes
2. **Falls through** to `return await super().readinto(b)` - fills buffer with NEXT N bytes
3. Returns the byte count from the second read, not the first

Result: **Data corruption** - buffer contains wrong bytes, and file position is advanced twice as far as expected.

**Bug 2: Wrong method in `readinto1()`** (_tempfile.py lines 372-377):

```python
async def readinto1(self: SpooledTemporaryFile[bytes], b: WriteableBuffer) -> int:
    if not self._rolled:
        await checkpoint_if_cancelled()
        self._fp.readinto(b)  # BUG 1: Missing 'return'
                              # BUG 2: Calls readinto instead of readinto1!

    return await super().readinto1(b)
```

Same missing return bug, PLUS calls `readinto()` instead of `readinto1()`, which has different semantics (readinto1 reads at most one buffer-size chunk without blocking).

**Impact:**
- **Silent data corruption**: Users get wrong data with no error
- **File position corruption**: File pointer is advanced incorrectly
- **Incorrect return values**: Returned byte count is wrong
- **High severity**: Core I/O functionality is broken for in-memory mode

## Fix

```diff
--- a/anyio/_core/_tempfile.py
+++ b/anyio/_core/_tempfile.py
@@ -365,14 +365,14 @@ class SpooledTemporaryFile(AsyncFile[AnyStr]):
     async def readinto(self: SpooledTemporaryFile[bytes], b: WriteableBuffer) -> int:
         if not self._rolled:
             await checkpoint_if_cancelled()
-            self._fp.readinto(b)
+            return self._fp.readinto(b)

         return await super().readinto(b)

     async def readinto1(self: SpooledTemporaryFile[bytes], b: WriteableBuffer) -> int:
         if not self._rolled:
             await checkpoint_if_cancelled()
-            self._fp.readinto(b)
+            return self._fp.readinto1(b)

         return await super().readinto1(b)
```