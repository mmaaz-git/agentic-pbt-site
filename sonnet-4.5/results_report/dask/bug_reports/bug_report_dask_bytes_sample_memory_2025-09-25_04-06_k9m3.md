# Bug Report: dask.bytes.read_bytes Sample Memory Exhaustion

**Target**: `dask.bytes.core.read_bytes`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `read_bytes` is called with both a `sample` parameter and a `delimiter` parameter, and the file does not contain the delimiter, the function reads the **entire file** into memory instead of limiting the sample to the requested size. This can cause memory exhaustion on large files.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
import tempfile
import os
from dask.bytes.core import read_bytes

@given(
    file_size=st.integers(min_value=1000, max_value=50000),
    sample_size=st.integers(min_value=100, max_value=500)
)
@settings(max_examples=50)
def test_sample_with_delimiter_reads_entire_file_if_no_delimiter(file_size, sample_size):
    content = b'x' * file_size
    assume(sample_size < file_size)

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(content)
        f.flush()
        temp_path = f.name

    try:
        sample, blocks = read_bytes(temp_path, blocksize=None, sample=sample_size, delimiter=b'\n')

        print(f"File size: {file_size}, Sample requested: {sample_size}, Actual sample: {len(sample)}")
        assert len(sample) <= sample_size * 2, \
            f"Sample exceeded reasonable size: requested {sample_size}, got {len(sample)}"
    finally:
        os.unlink(temp_path)
```

**Failing input**: File with 37,891 bytes, no newlines, sample_size=100

## Reproducing the Bug

```python
import tempfile
from dask.bytes.core import read_bytes

file_size = 10_000_000
sample_size = 100
content = b'x' * file_size

with tempfile.NamedTemporaryFile(delete=False) as f:
    f.write(content)
    temp_path = f.name

sample, blocks = read_bytes(temp_path, sample=sample_size, delimiter=b'\n')

print(f"File size: {file_size:,} bytes")
print(f"Requested sample size: {sample_size} bytes")
print(f"Actual sample size: {len(sample):,} bytes")

import os
os.unlink(temp_path)
```

Output:
```
File size: 10,000,000 bytes
Requested sample size: 100 bytes
Actual sample size: 10,000,000 bytes
```

## Why This Is A Bug

The documentation for `read_bytes` states that `sample` is "Whether or not to return a header sample" with values like `"1 MiB"` to control sample size. Users expect that specifying `sample=100` will read approximately 100 bytes, not the entire file.

The bug is in lines 173-184 of `dask/bytes/core.py`. When a delimiter is specified, the code enters a loop that keeps reading chunks until it finds the delimiter:

```python
sample_buff = f.read(sample)
while True:
    new = f.read(sample)
    if not new:
        break
    if delimiter in new:
        sample_buff = sample_buff + new.split(delimiter, 1)[0] + delimiter
        break
    sample_buff = sample_buff + new
```

If the delimiter never appears, this loop reads the **entire file** into `sample_buff`, violating the user's expectation that `sample` limits memory usage.

This is particularly dangerous for large files (e.g., multi-GB files without newlines), where it could cause out-of-memory errors.

## Fix

Limit the total bytes read when searching for a delimiter. After reading a reasonable multiple of the requested sample size without finding the delimiter, stop and return what has been read:

```diff
--- a/dask/bytes/core.py
+++ b/dask/bytes/core.py
@@ -170,14 +170,18 @@ def read_bytes(
             if delimiter is None:
                 sample = f.read(sample)
             else:
+                max_sample_bytes = sample * 10
+                bytes_read = 0
                 sample_buff = f.read(sample)
+                bytes_read += len(sample_buff)
                 while True:
                     new = f.read(sample)
-                    if not new:
+                    bytes_read += len(new)
+                    if not new or bytes_read >= max_sample_bytes:
                         break
                     if delimiter in new:
                         sample_buff = (
                             sample_buff + new.split(delimiter, 1)[0] + delimiter
                         )
                         break
                     sample_buff = sample_buff + new
                 sample = sample_buff
```

This fix caps the sample at 10x the requested size when searching for a delimiter, preventing unbounded memory usage while still allowing reasonable flexibility to find the delimiter.