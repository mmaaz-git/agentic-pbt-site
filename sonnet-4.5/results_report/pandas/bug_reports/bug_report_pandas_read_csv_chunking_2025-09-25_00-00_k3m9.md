# Bug Report: pandas.io.parsers read_csv Chunked Reading Type Inference Inconsistency

**Target**: `pandas.io.parsers.read_csv`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When reading a CSV file with `chunksize` parameter, pandas infers column types independently for each chunk, leading to inconsistent type inference compared to reading the entire file at once. This causes silent data corruption where the same CSV data produces different values depending on whether it's read in chunks or all at once.

## Property-Based Test

```python
import pandas as pd
import io
from hypothesis import given, strategies as st, settings
import hypothesis.extra.pandas as pdst


@given(
    df=pdst.data_frames(
        columns=[
            pdst.column('a', dtype=int),
            pdst.column('b', dtype=float),
            pdst.column('c', dtype=str),
        ],
        rows=st.tuples(
            st.integers(min_value=-1000, max_value=1000),
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            st.text(alphabet=st.characters(blacklist_categories=('Cs',)), min_size=0, max_size=20)
        )
    )
)
@settings(max_examples=100)
def test_chunking_equivalence(df):
    if len(df) == 0:
        return

    csv_string = df.to_csv(index=False)

    full_read = pd.read_csv(io.StringIO(csv_string))

    chunksize = max(1, len(df) // 3)
    chunks = []
    for chunk in pd.read_csv(io.StringIO(csv_string), chunksize=chunksize):
        chunks.append(chunk)

    chunked_read = pd.concat(chunks, ignore_index=True)

    pd.testing.assert_frame_equal(full_read, chunked_read)
```

**Failing input**: DataFrame with column containing values `['0', ':']` where the first value can be parsed as int but the full column cannot.

## Reproducing the Bug

```python
import pandas as pd
import io

csv_data = "a,b,c\n0,0.0,0\n0,0.0,:\n"

full_read = pd.read_csv(io.StringIO(csv_data))
chunked_read = pd.concat([chunk for chunk in pd.read_csv(io.StringIO(csv_data), chunksize=1)], ignore_index=True)

print("Full read column 'c':", full_read['c'].values)
print("Chunked read column 'c':", chunked_read['c'].values)
print("Equal?", (full_read['c'].values == chunked_read['c'].values).all())
```

**Output:**
```
Full read column 'c': ['0' ':']
Chunked read column 'c': [0 ':']
Equal? False
```

## Why This Is A Bug

This violates the fundamental contract of chunked reading: processing data in chunks should produce identical results to processing it all at once. The `chunksize` parameter exists specifically to handle large files that don't fit in memory, and users expect to get the same results regardless of chunk size.

The bug occurs because:
1. When reading the first chunk (row with "0"), pandas infers column `c` as `int64`
2. When reading the second chunk (row with ":"), pandas infers column `c` as `object`
3. After concatenation, the first value is `0` (int) while in full read it's `'0'` (string)
4. The full file read correctly infers `object` type by examining all rows

This is a **high severity** bug because:
- It causes **silent data corruption** - different values without any warning or error
- It affects **core functionality** - CSV reading is one of pandas' most critical features
- It's easily triggered in **real-world scenarios** - large files processed in chunks
- Users have no way to detect this issue without comparing chunked vs full reads

## Fix

The root cause is that each chunk performs independent type inference. The fix should ensure that type inference considers all chunks, or at a minimum, uses the most conservative (widest) type across chunks.

Potential approaches:
1. **Two-pass reading for chunked mode**: First pass infers types from entire file, second pass reads with explicit dtypes
2. **Progressive type widening**: Start with inferred type from first chunk, widen types as needed in subsequent chunks
3. **Document limitation**: If fixing is impractical, document that chunked reading may produce different types and recommend explicit `dtype` parameter

The most robust fix would be option 2 - maintaining type inference state across chunks and widening types (e.g., int → object) when a chunk contains values incompatible with the current inferred type.