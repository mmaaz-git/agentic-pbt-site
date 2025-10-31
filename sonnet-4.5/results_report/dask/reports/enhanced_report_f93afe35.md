# Bug Report: dask.bytes.read_bytes Delimiter Sampling Edge Case

**Target**: `dask.bytes.core.read_bytes`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `read_bytes` function incorrectly returns the entire file content as the sample when using both a delimiter and sample parameters, if the file is small enough to fit within the initial read operation. The sample should stop at the first delimiter boundary but instead includes data after it.

## Property-Based Test

```python
import tempfile
from pathlib import Path

from hypothesis import given, settings, strategies as st

from dask.bytes.core import read_bytes


@given(
    st.binary(min_size=1, max_size=100),
    st.binary(min_size=1, max_size=100),
    st.integers(min_value=50, max_value=200)
)
@settings(max_examples=500)
def test_delimiter_sampling_always_ends_with_delimiter(data_before, data_after, sample_size):
    delimiter = b'\n'

    if delimiter in data_before or delimiter in data_after:
        return

    full_data = data_before + delimiter + data_after

    with tempfile.NamedTemporaryFile(delete=False, mode='wb') as f:
        f.write(full_data)
        f.flush()
        temp_path = f.name

    try:
        sample, blocks = read_bytes(
            temp_path,
            blocksize=None,
            delimiter=delimiter,
            sample=sample_size
        )

        assert sample.endswith(delimiter), \
            f"Sample must end with delimiter. Got: {sample!r}"

    finally:
        Path(temp_path).unlink()

if __name__ == "__main__":
    test_delimiter_sampling_always_ends_with_delimiter()
```

<details>

<summary>
**Failing input**: `data_before=b'\x00'`, `data_after=b'\x00'`, `sample_size=50`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/5/hypo.py", line 43, in <module>
    test_delimiter_sampling_always_ends_with_delimiter()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/5/hypo.py", line 10, in test_delimiter_sampling_always_ends_with_delimiter
    st.binary(min_size=1, max_size=100),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/5/hypo.py", line 36, in test_delimiter_sampling_always_ends_with_delimiter
    assert sample.endswith(delimiter), \
           ~~~~~~~~~~~~~~~^^^^^^^^^^^
AssertionError: Sample must end with delimiter. Got: b'\x00\n\x00'
Falsifying example: test_delimiter_sampling_always_ends_with_delimiter(
    data_before=b'\x00',
    data_after=b'\x00',
    sample_size=50,  # or any other generated value
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/5/hypo.py:21
        /home/npc/miniconda/lib/python3.13/functools.py:54
        /home/npc/miniconda/lib/python3.13/tempfile.py:498
```
</details>

## Reproducing the Bug

```python
import tempfile
from pathlib import Path

from dask.bytes.core import read_bytes

data = b'\x00\n\x00'

with tempfile.NamedTemporaryFile(delete=False, mode='wb') as f:
    f.write(data)
    temp_path = f.name

sample, blocks = read_bytes(temp_path, blocksize=None, delimiter=b'\n', sample=100)

print(f"File content: {data!r}")
print(f"Sample: {sample!r}")
print(f"Sample ends with delimiter: {sample.endswith(b'\\n')}")

Path(temp_path).unlink()
```

<details>

<summary>
Output shows sample incorrectly includes data after delimiter
</summary>
```
File content: b'\x00\n\x00'
Sample: b'\x00\n\x00'
Sample ends with delimiter: False
```
</details>

## Why This Is A Bug

This behavior violates the expected contract of `read_bytes` when both `sample` and `delimiter` parameters are specified. The existing test suite in `dask/bytes/tests/test_local.py` explicitly validates that samples must end with the delimiter:

```python
def test_read_bytes_sample_delimiter():
    with filetexts(files, mode="b"):
        sample, values = read_bytes(".test.accounts.*", sample=80, delimiter=b"\n")
        assert sample.endswith(b"\n")  # Line 83 - expects delimiter at end
```

The documentation states that delimiters are used to "cleanly break data" with boundaries that "end on the delimiter" (lines 32-33 of core.py). When combined with the sample parameter described as a "header sample", the logical expectation is that the sample should respect delimiter boundaries.

The bug occurs because the implementation at lines 173-184 of `core.py` only checks for the delimiter in subsequent reads (`new` variable at line 178), but never checks the initial read stored in `sample_buff`. When a small file fits entirely within the first read at line 173, the while loop never executes (line 176 breaks immediately), causing the entire file content to be returned as the sample instead of stopping at the first delimiter.

## Relevant Context

The bug manifests specifically when:
- Both `sample` and `delimiter` parameters are provided to `read_bytes`
- The file is small enough that the content up to (and including) the first delimiter fits within the initial sample read
- The delimiter exists in the initial read buffer

This edge case is particularly relevant for:
- Unit tests that use small test files
- Processing small configuration or metadata files
- Development environments with sample data

The dask library documentation for `read_bytes` can be found at: https://docs.dask.org/en/stable/generated/dask.bytes.core.read_bytes.html

The affected code is in `/dask/bytes/core.py` at lines 173-184.

## Proposed Fix

```diff
--- a/dask/bytes/core.py
+++ b/dask/bytes/core.py
@@ -172,6 +172,10 @@ def read_bytes(
             else:
                 sample_buff = f.read(sample)
+                # Check if delimiter exists in the initial read
+                if delimiter in sample_buff:
+                    sample_buff = sample_buff.split(delimiter, 1)[0] + delimiter
+                else:
                 while True:
                     new = f.read(sample)
                     if not new:
```