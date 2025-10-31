# Bug Report: dask.dataframe String Dtype Not Preserved in Round-Trip

**Target**: `dask.dataframe.from_pandas()`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

When converting a pandas DataFrame with object-dtype string columns to dask and back via `from_pandas().compute()`, the string columns are converted from `object` dtype to `string[pyarrow]` dtype, violating the round-trip property.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env')

import pandas as pd
import dask.dataframe as dd
from hypothesis import given, settings
import hypothesis.extra.pandas as pd_st


@given(pd_st.data_frames([
    pd_st.column('a', dtype=int),
    pd_st.column('b', dtype=float),
    pd_st.column('c', dtype=str),
]))
@settings(max_examples=100)
def test_from_pandas_round_trip(df):
    ddf = dd.from_pandas(df, npartitions=2)
    result = ddf.compute()

    pd.testing.assert_frame_equal(result, df, check_dtype=True)
```

**Failing input**:
```python
df = pd.DataFrame({
    'a': [1, 2, 3],
    'b': [1.0, 2.0, 3.0],
    'c': ['x', 'y', 'z']
})
```

## Reproducing the Bug

```python
import pandas as pd
import dask.dataframe as dd

df = pd.DataFrame({
    'a': [1, 2, 3],
    'b': [1.0, 2.0, 3.0],
    'c': ['x', 'y', 'z']
})

print("Original dtypes:")
print(df.dtypes)

ddf = dd.from_pandas(df, npartitions=2)
result = ddf.compute()

print("\nAfter round-trip dtypes:")
print(result.dtypes)

print(f"\nColumn 'c' dtype changed: {df['c'].dtype} -> {result['c'].dtype}")
assert df['c'].dtype == result['c'].dtype, f"Expected {df['c'].dtype}, got {result['c'].dtype}"
```

Output:
```
Original dtypes:
a      int64
b    float64
c     object
dtype: object

After round-trip dtypes:
a              int64
b            float64
c    string[pyarrow]
dtype: object

Column 'c' dtype changed: object -> string[pyarrow]
AssertionError: Expected object, got string[pyarrow]
```

## Why This Is A Bug

The `from_pandas` function should preserve the dtypes of the input DataFrame. While `string[pyarrow]` and `object` dtype may be functionally similar for string data, changing dtypes during a round-trip operation breaks the contract of preservation and can cause issues in downstream code that expects specific dtypes. This violates user expectations and the documented behavior that dask should mirror pandas.

Users may have code that depends on specific dtype checks (e.g., `df.dtypes == 'object'`), or may serialize the data expecting consistent dtypes. The automatic conversion can cause subtle bugs in production pipelines.

## Fix

Dask should provide an option to preserve the original pandas dtypes when converting from pandas, or at minimum should document this behavior clearly. The function `from_pandas` appears to be using PyArrow for string handling optimization, but this should be opt-in rather than automatic to preserve backward compatibility and round-trip guarantees.