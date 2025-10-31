# Bug Report: pandas.api.interchange Silently Converts NA to False in Boolean Columns

**Target**: `pandas.api.interchange.from_dataframe`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `from_dataframe` function silently converts NA values to False in boolean columns, causing silent data corruption.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
from pandas.api.interchange import from_dataframe
from hypothesis import given, strategies as st, settings


@given(st.lists(st.booleans(), min_size=1, max_size=20))
@settings(max_examples=50)
def test_nullable_bool_dtype(values):
    df = pd.DataFrame({'col': pd.array(values, dtype='boolean')})

    interchange_obj = df.__dataframe__()
    result = from_dataframe(interchange_obj)

    pd.testing.assert_frame_equal(result, df)
```

**Failing input**: Any DataFrame with nullable boolean dtype containing NA values

## Reproducing the Bug

```python
import pandas as pd
from pandas.api.interchange import from_dataframe

df = pd.DataFrame({'col': pd.array([True, False, None], dtype='boolean')})
print("Original:")
print(df)
print("Has NA?", df['col'].isna().any())

result = from_dataframe(df.__dataframe__())
print("\nAfter round-trip:")
print(result)
print("Has NA?", result['col'].isna().any())
```

Output:
```
Original:
     col
0   True
1  False
2   <NA>
Has NA? True

After round-trip:
     col
0   True
1  False
2  False
Has NA? False
```

## Why This Is A Bug

This is a **silent data corruption** bug. The interchange protocol converts NA (missing) values to False without any warning or error. This violates:

1. **Data integrity**: Missing data should remain missing, not be silently converted to a value
2. **Round-trip property**: `from_dataframe(df.__dataframe__())` should preserve all data
3. **User expectations**: No error is raised, so users may not notice their data has been corrupted

This is particularly dangerous because:
- The conversion happens silently (no error or warning)
- False is a valid boolean value, so the corruption is not obvious
- Users may make incorrect decisions based on corrupted data

## Fix

The fix should preserve NA values in boolean columns. The interchange protocol should either:
1. Convert to nullable boolean dtype on the result
2. Preserve the NA values in some other way
3. Raise an error if the round-trip cannot preserve the data

The root cause is likely in the buffer handling for boolean columns. The interchange protocol may not be properly handling the validity buffer that indicates which values are NA.