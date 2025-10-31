# Bug Report: dask.dataframe.io.orc Index Preservation in Multi-Partition Write/Read

**Target**: `dask.dataframe.io.orc` (specifically `to_orc` and `read_orc`)
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When writing a Dask DataFrame to ORC files with multiple partitions using `write_index=True`, and then reading it back, the index values are not preserved correctly. Each partition's index is reset to start at 0, causing silent data corruption.

## Property-Based Test

```python
import tempfile
import shutil
import pandas as pd
import dask.dataframe as dd
from hypothesis import given, settings, strategies as st
from hypothesis.extra.pandas import data_frames, column, range_indexes


@given(
    data_frames(
        columns=[
            column("int_col", dtype=int, elements=st.integers(min_value=-1000, max_value=1000)),
            column("float_col", dtype=float, elements=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)),
            column("str_col", dtype=str, elements=st.text(min_size=0, max_size=20)),
        ],
        index=range_indexes(min_size=1, max_size=100),
    )
)
@settings(max_examples=50, deadline=None)
def test_orc_round_trip_preserves_data(pdf):
    tmpdir = tempfile.mkdtemp()
    try:
        ddf = dd.from_pandas(pdf, npartitions=2)
        orc_path = f"{tmpdir}/test_orc"
        ddf.to_orc(orc_path, write_index=True)
        result_ddf = dd.read_orc(orc_path)
        result_pdf = result_ddf.compute()
        pd.testing.assert_frame_equal(pdf, result_pdf, check_dtype=False)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
```

**Failing input**:
```python
pdf = pd.DataFrame({
    'int_col': [0, 0],
    'float_col': [0.0, 0.0],
    'str_col': ['', '']
})
```

## Reproducing the Bug

```python
import tempfile
import shutil
import pandas as pd
import dask.dataframe as dd

tmpdir = tempfile.mkdtemp()
try:
    pdf = pd.DataFrame({
        'int_col': [0, 1, 2, 3],
        'float_col': [0.0, 1.0, 2.0, 3.0],
    })

    print(f"Original index: {list(pdf.index)}")

    ddf = dd.from_pandas(pdf, npartitions=2)
    orc_path = f"{tmpdir}/test_orc"
    ddf.to_orc(orc_path, write_index=True)

    result_ddf = dd.read_orc(orc_path)
    result_pdf = result_ddf.compute()

    print(f"Result index:   {list(result_pdf.index)}")

    assert list(pdf.index) == list(result_pdf.index), f"Index mismatch: {list(pdf.index)} != {list(result_pdf.index)}"
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)
```

**Output:**
```
Original index: [0, 1, 2, 3]
Result index:   [0, 1, 0, 1]
AssertionError: Index mismatch: [0, 1, 2, 3] != [0, 1, 0, 1]
```

Even worse with custom index values:
```python
pdf = pd.DataFrame({
    'int_col': [10, 20, 30, 40],
}, index=[100, 200, 300, 400])

ddf = dd.from_pandas(pdf, npartitions=2)
ddf.to_orc(orc_path, write_index=True)
result = dd.read_orc(orc_path).compute()

print(f"Original: {list(pdf.index)}")
print(f"Result:   {list(result.index)}")
```

**Output:**
```
Original: [100, 200, 300, 400]
Result:   [0, 1, 0, 1]
```

## Why This Is A Bug

1. **Violates round-trip property**: Writing a DataFrame to ORC and reading it back should preserve the data, including the index.

2. **Violates documented behavior**: The `write_index=True` parameter (which is the default) claims to write the index, but the index values are not correctly preserved when reading back.

3. **Silent data corruption**: The bug causes incorrect index values without raising any errors, which can lead to incorrect joins, merges, and other operations that rely on index values.

4. **Only affects multi-partition writes**: Single partition writes work correctly, making this bug subtle and hard to detect.

## Fix

The root cause is that when reading ORC files back, each partition's index is being reset independently. The fix likely involves either:

1. Storing the original index offset for each partition in the ORC metadata, or
2. Using the partition boundaries to reconstruct the correct index values when reading

This would require changes to both `to_orc` (to store index metadata) and `read_orc` (to reconstruct the correct index from metadata).

Since this is a complex fix involving the ORC engine's metadata handling, a detailed patch would require deeper investigation of the `_read_orc` function and the engine's `read_metadata` and `write_partition` methods.