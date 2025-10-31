# Bug Report: pandas.io.json Series dtype coercion in round-trip

**Target**: `pandas.io.json.read_json` / `pandas.Series.to_json`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

When round-tripping a Series with float64 dtype through JSON serialization, integer-valued floats are coerced to int64, violating the dtype preservation property expected from round-trip operations.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import io
import pandas as pd
from pandas import Series

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
                min_size=1, max_size=50))
@settings(max_examples=200)
def test_series_roundtrip_index(data):
    s = Series(data)
    json_str = s.to_json(orient='index')
    result = pd.read_json(io.StringIO(json_str), typ='series', orient='index')
    pd.testing.assert_series_equal(s, result)
```

**Failing input**: `data=[0.0]`

## Reproducing the Bug

```python
import io
import pandas as pd
from pandas import Series

s = Series([0.0], dtype='float64')
print(f"Original dtype: {s.dtype}")

json_str = s.to_json(orient='index')
result = pd.read_json(io.StringIO(json_str), typ='series', orient='index')
print(f"Result dtype: {result.dtype}")

print(f"Bug present: {s.dtype != result.dtype}")
```

Output:
```
Original dtype: float64
Result dtype: int64
Bug present: True
```

## Why This Is A Bug

The `read_json` documentation includes round-trip examples (lines 710-743 in `_json.py`) showing that `pd.read_json(StringIO(df.to_json(orient=X)), orient=X)` should recover the original DataFrame or Series. While the examples don't explicitly claim dtype preservation, they strongly imply it by presenting the round-trip as a way to encode/decode data structures.

The issue occurs because `read_json` has `dtype=True` by default, which enables dtype inference. When all float values happen to be integer-valued (like `0.0`, `1.0`, `2.0`), pandas infers them as `int64` rather than preserving the original `float64` dtype.

This violates users' reasonable expectations that:
```python
s == pd.read_json(io.StringIO(s.to_json(orient='index')), typ='series', orient='index')
```

The behavior affects real-world usage where users may serialize numerical data with specific dtypes (e.g., for consistency with C libraries or for memory-mapped arrays) and expect those dtypes to be preserved.

## Fix

The documentation should either:

1. Prominently document in the `read_json` examples that `dtype=False` should be used for true round-trip dtype preservation:

```diff
 >>> df.to_json(orient='split')
     '{{"columns":["col 1","col 2"],"index":["row 1","row 2"],"data":[["a","b"],["c","d"]]}}'
->>> pd.read_json(StringIO(_), orient='split')
+>>> pd.read_json(StringIO(_), orient='split', dtype=False)
       col 1 col 2
 row 1     a     b
 row 2     c     d
```

OR

2. Change the default behavior to preserve dtypes when round-tripping from pandas objects. This could be done by detecting when the JSON was produced by pandas (e.g., via a metadata field) and disabling dtype inference in that case.

OR

3. Add a `Series.from_json()` / `DataFrame.from_json()` classmethod that guarantees round-trip compatibility:

```python
@classmethod
def from_json(cls, json_str, orient=None):
    """Read JSON produced by to_json() with dtype preservation."""
    return read_json(io.StringIO(json_str), typ='series', orient=orient, dtype=False)
```

The simplest fix is option 1: update documentation to show `dtype=False` in all round-trip examples.