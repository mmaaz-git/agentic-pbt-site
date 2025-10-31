# Bug Report: anyio SpooledTemporaryFile readinto() Double-Read Data Corruption

**Target**: `anyio.SpooledTemporaryFile.readinto()` and `anyio.SpooledTemporaryFile.readinto1()`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `readinto()` and `readinto1()` methods in anyio's SpooledTemporaryFile have missing return statements causing double-read data corruption when the file is in memory mode, resulting in wrong data being read and incorrect file position advancement.

## Property-Based Test

```python
import pytest
import anyio
from anyio import SpooledTemporaryFile
from hypothesis import given, strategies as st, settings


@settings(max_examples=100)
@given(
    data=st.binary(min_size=1, max_size=100),
    buffer_size=st.integers(min_value=1, max_value=50)
)
@pytest.mark.anyio
async def test_readinto_reads_correct_data(data, buffer_size):
    """
    Property: readinto() should read data from the current position into the buffer
    and return the number of bytes read. The buffer should contain the data from
    the file starting at the current position.
    """
    async with SpooledTemporaryFile(max_size=1000, mode='w+b') as f:
        # Write test data
        await f.write(data)
        await f.seek(0)

        # Create buffer no larger than data
        actual_buffer_size = min(buffer_size, len(data))
        buffer = bytearray(actual_buffer_size)

        # Read into buffer
        bytes_read = await f.readinto(buffer)

        # Property checks
        assert bytes_read == actual_buffer_size, f"Should read {actual_buffer_size} bytes, but read {bytes_read}"
        expected_data = data[:actual_buffer_size]
        assert bytes(buffer) == expected_data, f"Buffer should contain {expected_data!r}, but contains {bytes(buffer)!r}"

        # Check file position
        position = await f.tell()
        assert position == actual_buffer_size, f"File position should be {actual_buffer_size}, but is {position}"


if __name__ == "__main__":
    # Run a simple test case to demonstrate the bug
    import asyncio

    async def run_test():
        try:
            await test_readinto_reads_correct_data(b'0123456789', 5)
            print("Test passed!")
        except AssertionError as e:
            print(f"Test failed: {e}")

    asyncio.run(run_test())
```

<details>

<summary>
**Failing input**: `data=b'\x00\x01', buffer_size=1`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/50
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 2 items

hypo.py::test_readinto_reads_correct_data[asyncio] FAILED

=================================== FAILURES ===================================
__________________ test_readinto_reads_correct_data[asyncio] ___________________
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 8, in test_readinto_reads_correct_data
  |     @given(
  |
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 3 distinct failures. (3 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/anyio/pytest_plugin.py", line 140, in run_with_hypothesis
    |     runner.run_test(original_func, kwargs)
    |     ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/anyio/_backends/_asyncio.py", line 2279, in run_test
    |     self._raise_async_exceptions()
    |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/anyio/_backends/_asyncio.py", line 2183, in _raise_async_exceptions
    |     raise exceptions[0]
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/anyio/_backends/_asyncio.py", line 2273, in run_test
    |     self.get_loop().run_until_complete(
    |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
    |         self._call_in_runner_task(test_func, **kwargs)
    |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |     )
    |     ^
    |   File "/home/npc/miniconda/lib/python3.13/asyncio/base_events.py", line 725, in run_until_complete
    |     return future.result()
    |            ~~~~~~~~~~~~~^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/anyio/_backends/_asyncio.py", line 2233, in _call_in_runner_task
    |     return await future
    |            ^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/anyio/_backends/_asyncio.py", line 2200, in _run_tests_and_fixtures
    |     retval = await coro
    |              ^^^^^^^^^^
    |   File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 34, in test_readinto_reads_correct_data
    |     assert bytes(buffer) == expected_data, f"Buffer should contain {expected_data!r}, but contains {bytes(buffer)!r}"
    | AssertionError: Buffer should contain b'\x00', but contains b'\x01'
    | assert b'\x01' == b'\x00'
    |
    |   At index 0 diff: b'\x01' != b'\x00'
    |
    |   Full diff:
    |   - b'\x00'
    |   ?      ^
    |   + b'\x01'
    |   ?      ^
    | Falsifying example: run_with_hypothesis(
    |     data=b'\x00\x01',  # or any other generated value
    |     buffer_size=1,
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/anyio/pytest_plugin.py", line 140, in run_with_hypothesis
    |     runner.run_test(original_func, kwargs)
    |     ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/anyio/_backends/_asyncio.py", line 2279, in run_test
    |     self._raise_async_exceptions()
    |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/anyio/_backends/_asyncio.py", line 2183, in _raise_async_exceptions
    |     raise exceptions[0]
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/anyio/_backends/_asyncio.py", line 2273, in run_test
    |     self.get_loop().run_until_complete(
    |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
    |         self._call_in_runner_task(test_func, **kwargs)
    |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |     )
    |     ^
    |   File "/home/npc/miniconda/lib/python3.13/asyncio/base_events.py", line 725, in run_until_complete
    |     return future.result()
    |            ~~~~~~~~~~~~~^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/anyio/_backends/_asyncio.py", line 2233, in _call_in_runner_task
    |     return await future
    |            ^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/anyio/_backends/_asyncio.py", line 2200, in _run_tests_and_fixtures
    |     retval = await coro
    |              ^^^^^^^^^^
    |   File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 38, in test_readinto_reads_correct_data
    |     assert position == actual_buffer_size, f"File position should be {actual_buffer_size}, but is {position}"
    | AssertionError: File position should be 1, but is 2
    | assert 2 == 1
    | Falsifying example: run_with_hypothesis(
    |     data=b'\x00\x00',
    |     buffer_size=1,
    | )
    +---------------- 3 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/anyio/pytest_plugin.py", line 140, in run_with_hypothesis
    |     runner.run_test(original_func, kwargs)
    |     ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/anyio/_backends/_asyncio.py", line 2279, in run_test
    |     self._raise_async_exceptions()
    |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/anyio/_backends/_asyncio.py", line 2183, in _raise_async_exceptions
    |     raise exceptions[0]
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/anyio/_backends/_asyncio.py", line 2273, in run_test
    |     self.get_loop().run_until_complete(
    |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
    |         self._call_in_runner_task(test_func, **kwargs)
    |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |     )
    |     ^
    |   File "/home/npc/miniconda/lib/python3.13/asyncio/base_events.py", line 725, in run_until_complete
    |     return future.result()
    |            ~~~~~~~~~~~~~^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/anyio/_backends/_asyncio.py", line 2233, in _call_in_runner_task
    |     return await future
    |            ^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/anyio/_backends/_asyncio.py", line 2200, in _run_tests_and_fixtures
    |     retval = await coro
    |              ^^^^^^^^^^
    |   File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 32, in test_readinto_reads_correct_data
    |     assert bytes_read == actual_buffer_size, f"Should read {actual_buffer_size} bytes, but read {bytes_read}"
    | AssertionError: Should read 1 bytes, but read 0
    | assert 0 == 1
    | Falsifying example: run_with_hypothesis(
    |     data=b'\x00',
    |     buffer_size=1,  # or any other generated value
    | )
    +------------------------------------
=========================== short test summary info ============================
FAILED hypo.py::test_readinto_reads_correct_data[asyncio] - ExceptionGroup: H...
!!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!!!!!!!!
============================== 1 failed in 1.74s ===============================
```
</details>

## Reproducing the Bug

```python
import anyio
from anyio import SpooledTemporaryFile


async def demonstrate_bug():
    async with SpooledTemporaryFile(max_size=1000, mode='w+b') as f:
        # Write test data
        await f.write(b'0123456789ABCDEF')
        await f.seek(0)

        # Try to read first 5 bytes into buffer
        buffer = bytearray(5)
        bytes_read = await f.readinto(buffer)

        print(f"Bytes read: {bytes_read}")
        print(f"Buffer contents: {buffer}")
        print(f"Expected: b'01234' (first 5 bytes)")
        print(f"Actual: {bytes(buffer)}")

        # Check file position after read
        position = await f.tell()
        print(f"File position after readinto: {position}")
        print(f"Expected position: 5")

        if bytes(buffer) != b'01234':
            print("\nBUG CONFIRMED: Data is offset! We got wrong bytes instead of 0-4")
        else:
            print("\nNo bug detected")


# Run the demonstration
anyio.run(demonstrate_bug)
```

<details>

<summary>
Output showing double-read corruption
</summary>
```
Bytes read: 5
Buffer contents: bytearray(b'56789')
Expected: b'01234' (first 5 bytes)
Actual: b'56789'
File position after readinto: 10
Expected position: 5

BUG CONFIRMED: Data is offset! We got wrong bytes instead of 0-4
```
</details>

## Why This Is A Bug

The bug occurs in the `readinto()` and `readinto1()` methods of SpooledTemporaryFile when the file is in memory mode (not rolled to disk). The issue is caused by **missing return statements** that result in the methods executing twice:

1. **Double-read execution**: When `not self._rolled` is true, the method calls `self._fp.readinto(b)` which correctly reads data into the buffer. However, because there's no `return` statement, execution continues to the next line which calls `await super().readinto(b)`, reading AGAIN and overwriting the buffer with the next N bytes.

2. **Wrong data returned**: The buffer ends up containing bytes 5-9 instead of bytes 0-4 in our example, because the first read consumed bytes 0-4 and the second read consumed bytes 5-9.

3. **File position corruption**: The file position advances twice as far as expected (10 bytes instead of 5), making subsequent reads start from the wrong position.

4. **Wrong method call in readinto1()**: The `readinto1()` method incorrectly calls `self._fp.readinto(b)` instead of `self._fp.readinto1(b)`, violating the semantic difference between these methods (readinto1 should perform at most one read system call).

This violates the Python I/O protocol documented in the [io module](https://docs.python.org/3/library/io.html#io.BufferedIOBase.readinto), which states that `readinto()` should "Read bytes into a pre-allocated, writable bytes-like object b and return the number of bytes read."

## Relevant Context

The bug is located in `/home/npc/pbt/agentic-pbt/envs/anyio_env/lib/python3.13/site-packages/anyio/_core/_tempfile.py` at lines 365-377:

```python
async def readinto(self: SpooledTemporaryFile[bytes], b: WriteableBuffer) -> int:
    if not self._rolled:
        await checkpoint_if_cancelled()
        self._fp.readinto(b)  # Missing return statement!

    return await super().readinto(b)  # This always executes, causing double-read

async def readinto1(self: SpooledTemporaryFile[bytes], b: WriteableBuffer) -> int:
    if not self._rolled:
        await checkpoint_if_cancelled()
        self._fp.readinto(b)  # Missing return AND wrong method (should be readinto1)!

    return await super().readinto1(b)
```

This affects all uses of SpooledTemporaryFile in binary mode when the file hasn't been rolled to disk (i.e., when the content size is below `max_size`). The bug is particularly dangerous because it causes **silent data corruption** - the methods complete without error but return incorrect data.

Workaround: Users can use the `read()` method instead of `readinto()`, which works correctly.

## Proposed Fix

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