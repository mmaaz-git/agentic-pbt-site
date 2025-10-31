# Bug Report: dask.bytes.read_bytes Returns Integer Instead of Bytes When sample=0

**Target**: `dask.bytes.core.read_bytes`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

When `read_bytes` is called with `sample=0` (integer zero), it returns the integer `0` instead of an empty bytes object `b''`, violating the documented return type contract that promises bytes.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.bytes.core import read_bytes
import tempfile
import os


@given(st.integers(min_value=0, max_value=1000))
@settings(max_examples=200)
def test_sample_return_type_with_integer(sample_size):
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, 'test.txt')

        test_data = b'x' * 1000
        with open(test_file, 'wb') as f:
            f.write(test_data)

        sample, blocks = read_bytes(test_file, sample=sample_size, blocksize=None)

        assert isinstance(sample, bytes), \
            f"sample={sample_size} should return bytes, got {type(sample).__name__}"


if __name__ == "__main__":
    test_sample_return_type_with_integer()
```

<details>

<summary>
**Failing input**: `sample_size=0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 24, in <module>
    test_sample_return_type_with_integer()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 8, in test_sample_return_type_with_integer
    @settings(max_examples=200)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 19, in test_sample_return_type_with_integer
    assert isinstance(sample, bytes), \
           ~~~~~~~~~~^^^^^^^^^^^^^^^
AssertionError: sample=0 should return bytes, got int
Falsifying example: test_sample_return_type_with_integer(
    sample_size=0,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/25/hypo.py:20
```
</details>

## Reproducing the Bug

```python
import tempfile
import os
from dask.bytes.core import read_bytes

with tempfile.TemporaryDirectory() as tmpdir:
    test_file = os.path.join(tmpdir, 'test.txt')
    with open(test_file, 'wb') as f:
        f.write(b'hello world')

    sample, blocks = read_bytes(test_file, sample=0, blocksize=None)

    print(f"Type: {type(sample)}")
    print(f"Value: {sample!r}")

    # This will fail with AttributeError since sample is int, not bytes
    try:
        sample.decode('utf-8')
        print("Successfully decoded as UTF-8")
    except AttributeError as e:
        print(f"AttributeError: {e}")
```

<details>

<summary>
Type Error: 'int' object has no attribute 'decode'
</summary>
```
Type: <class 'int'>
Value: 0
AttributeError: 'int' object has no attribute 'decode'
```
</details>

## Why This Is A Bug

The `read_bytes` function violates its documented return type contract. According to the docstring at `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/bytes/core.py:51-70`:

```python
Parameters
----------
sample : int, string, or boolean
    Whether or not to return a header sample.
    Values can be ``False`` for "no sample requested"
    Or an integer or string value like ``2**20`` or ``"1 MiB"``

Returns
-------
sample : bytes
    The sample header
```

The documentation explicitly states that:
1. The `sample` parameter accepts integers as valid input
2. The return value for `sample` is always of type `bytes`

However, when `sample=0` (an integer), the function returns the integer `0` instead of empty bytes `b''`. This inconsistency is problematic because:

1. **Type contract violation**: Code expecting bytes will fail with AttributeError when trying to call bytes methods like `.decode()`, `.split()`, etc.
2. **Inconsistent behavior**: The function behaves differently for semantically equivalent inputs:
   - `sample="0"` (string) correctly returns `b''` (empty bytes)
   - `sample=0` (integer) incorrectly returns `0` (integer)
3. **Falsy value confusion**: The bug occurs because line 163 uses `if sample:` which treats integer 0 as falsy, causing it to skip the sample processing entirely

## Relevant Context

The root cause is in the sample handling logic at line 163 of `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/bytes/core.py`:

```python
if sample:  # Line 163
    if sample is True:
        sample = "10 kiB"  # backwards compatibility
    if isinstance(sample, str):
        sample = parse_bytes(sample)
    with OpenFile(fs, paths[0], compression=compression) as f:
        # read block without seek (because we start at zero)
        if delimiter is None:
            sample = f.read(sample)
        else:
            # ... delimiter handling
```

The condition `if sample:` evaluates to `False` when `sample=0` because `bool(0) == False`. This causes the function to skip the entire sample processing block and return the unmodified integer value.

Testing confirms the inconsistent behavior:
- `sample=False` → returns `False` (bool) - expected "no sample" behavior
- `sample=0` (int) → returns `0` (int) - BUG: should return `b''`
- `sample=1` (int) → returns `b'h'` (bytes) - correct
- `sample="0"` (str) → returns `b''` (bytes) - correct
- `sample="1 B"` (str) → returns `b'h'` (bytes) - correct

## Proposed Fix

The fix requires changing the condition to explicitly check for `False` rather than relying on truthiness:

```diff
--- a/dask/bytes/core.py
+++ b/dask/bytes/core.py
@@ -160,7 +160,7 @@ def read_bytes(
         ]
         out.append(values)

-    if sample:
+    if sample is not False:
         if sample is True:
             sample = "10 kiB"  # backwards compatibility
         if isinstance(sample, str):
```

This fix ensures that:
- `sample=False` continues to mean "no sample requested" and returns `False`
- `sample=0` is treated as "read 0 bytes" and returns `b''`
- All other integer values continue to work correctly
- String values continue to work as before
- Backward compatibility is preserved