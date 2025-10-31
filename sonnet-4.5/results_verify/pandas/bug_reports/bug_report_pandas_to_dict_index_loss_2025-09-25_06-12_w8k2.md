# Bug Report: pandas DataFrame.to_dict index type loss on empty DataFrames

**Target**: `pandas.DataFrame.to_dict` / `pandas.DataFrame.from_dict`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

Empty DataFrames with RangeIndex lose their index type when round-tripped through `to_dict(orient='dict')` and `from_dict()`, converting from RangeIndex to Index with dtype='object' and inferred_type='empty'.

## Property-Based Test

```python
from hypothesis import given, settings
from hypothesis.extra.pandas import column, data_frames
import pandas as pd

@given(data_frames([
    column('a', dtype=int),
    column('b', dtype=float),
    column('c', dtype=str),
]))
@settings(max_examples=100)
def test_dataframe_to_dict_from_dict_roundtrip_dict_orient(df):
    result = pd.DataFrame.from_dict(df.to_dict(orient='dict'))
    pd.testing.assert_frame_equal(result, df)
```

**Failing input**: Empty DataFrame with RangeIndex

## Reproducing the Bug

```python
import pandas as pd

df = pd.DataFrame({'a': [], 'b': [], 'c': []})
df['a'] = df['a'].astype(int)
df['b'] = df['b'].astype(float)
df['c'] = df['c'].astype(str)

print("Original index:")
print(f"  Type: {type(df.index).__name__}")
print(f"  inferred_type: {df.index.inferred_type}")

d = df.to_dict(orient='dict')
result = pd.DataFrame.from_dict(d)

print("\nAfter round-trip:")
print(f"  Type: {type(result.index).__name__}")
print(f"  inferred_type: {result.index.inferred_type}")
```

Output:
```
Original index:
  Type: RangeIndex
  inferred_type: integer

After round-trip:
  Type: Index
  inferred_type: empty
```

## Why This Is A Bug

The `to_dict`/`from_dict` API should preserve DataFrame structure including index types. RangeIndex is semantically meaningful - it represents a range-based index which is memory-efficient. After round-tripping, the index becomes a generic Index with inferred_type='empty' instead of 'integer', violating the expectation that serialization preserves structure.

This affects users who serialize empty DataFrames and expect to restore them with the same index type.

## Fix

When `from_dict` receives an empty dictionary (orient='dict'), it should construct a RangeIndex instead of an empty Index. This would match the default behavior when creating empty DataFrames directly.