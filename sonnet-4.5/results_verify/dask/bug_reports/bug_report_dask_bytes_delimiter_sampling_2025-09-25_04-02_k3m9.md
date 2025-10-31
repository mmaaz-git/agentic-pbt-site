# Bug Report: dask.bytes.read_bytes Delimiter Sampling

**Target**: `dask.bytes.core.read_bytes`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When using `read_bytes` with a delimiter and sample size, if the entire file fits within the first read operation, the returned sample incorrectly includes data after the first delimiter instead of stopping at it.

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
```

**Failing input**: `data_before=b'\x00'`, `data_after=b'\x00'`, `sample_size=50`

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
print(f"Sample ends with delimiter: {sample.endswith(b'\n')}")

Path(temp_path).unlink()
```

Output:
```
File content: b'\x00\n\x00'
Sample: b'\x00\n\x00'
Sample ends with delimiter: False
```

Expected: `sample` should be `b'\x00\n'` (ending at the first delimiter).

## Why This Is A Bug

The existing test suite explicitly verifies this property in `dask/bytes/tests/test_local.py:83`:
```python
sample, values = read_bytes(".test.accounts.1.json", sample=80, delimiter=b"\n")
assert sample.endswith(b"\n")
```

The docstring describes the sample as a "header sample," which when combined with delimiter-based reading should logically stop at the first delimiter boundary. The code in lines 173-184 of `core.py` clearly attempts to implement this behavior by searching for the delimiter and stopping there.

However, the implementation has a logic error: it only checks for the delimiter in subsequent reads (`new`), not in the initial `sample_buff`. When the entire file fits in the first read, the delimiter check is never performed, causing the sample to include data beyond the first delimiter.

## Fix

```diff
--- a/dask/bytes/core.py
+++ b/dask/bytes/core.py
@@ -170,6 +170,13 @@ def read_bytes(
             if delimiter is None:
                 sample = f.read(sample)
             else:
                 sample_buff = f.read(sample)
+                # Check if delimiter is in the initial read
+                if delimiter in sample_buff:
+                    sample = sample_buff.split(delimiter, 1)[0] + delimiter
+                else:
+                    # Continue reading until delimiter is found
                 while True:
                     new = f.read(sample)
                     if not new:
```

Alternative minimal fix:
```diff
--- a/dask/bytes/core.py
+++ b/dask/bytes/core.py
@@ -172,6 +172,11 @@ def read_bytes(
             else:
                 sample_buff = f.read(sample)
+                if delimiter in sample_buff:
+                    sample_buff = sample_buff.split(delimiter, 1)[0] + delimiter
+                    sample = sample_buff
+                else:
                 while True:
                     new = f.read(sample)
                     if not new:
```