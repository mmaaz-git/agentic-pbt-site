# Bug Report: pandas.io.json Type coercion breaks round-trip property

**Target**: `pandas.io.json.read_json` / `pandas.Series.to_json` / `pandas.DataFrame.to_json`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

When round-tripping pandas Series and DataFrames through JSON serialization, both data types and index types are coerced due to aggressive type inference, violating the round-trip property implied by the documentation examples. This affects both the `dtype` parameter (data inference) and `convert_axes` parameter (index/column inference), both of which default to enabling type coercion.

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

**Failing inputs**:
- `data=[0.0]` (dtype changes from float64 to int64)
- Series with index `['0']` (index type changes from str to int64)

## Reproducing the Bug

### Bug 1: Data dtype coercion

```python
import io
import pandas as pd
from pandas import Series

s = Series([0.0], dtype='float64')
print(f"Original dtype: {s.dtype}")

json_str = s.to_json(orient='index')
result = pd.read_json(io.StringIO(json_str), typ='series', orient='index')
print(f"Result dtype: {result.dtype}")
```

Output:
```
Original dtype: float64
Result dtype: int64
```

### Bug 2: Index dtype coercion

```python
import io
import pandas as pd
from pandas import Series

s = Series([1], index=['0'])
print(f"Original index type: {type(s.index[0])}")

json_str = s.to_json(orient='index')
result = pd.read_json(io.StringIO(json_str), typ='series', orient='index')
print(f"Result index type: {type(result.index[0])}")
```

Output:
```
Original index type: <class 'str'>
Result index type: <class 'numpy.int64'>
```

## Why This Is A Bug

The `read_json` documentation includes round-trip examples (lines 710-743 in `_json.py`) showing that `pd.read_json(StringIO(df.to_json(orient=X)), orient=X)` should recover the original DataFrame or Series. While the examples don't explicitly claim dtype preservation, they strongly imply it by presenting the round-trip as a standard encode/decode pattern.

The issues occur because:
1. `read_json` has `dtype=True` by default, enabling dtype inference on data values
2. `read_json` has `convert_axes=True` by default (for non-table orients), enabling type inference on indices/columns

When all float values happen to be integer-valued (like `0.0`, `1.0`, `2.0`), pandas infers them as `int64` rather than preserving the original `float64` dtype. Similarly, when index labels are strings that look like numbers (like `'0'`, `'1'`), they are converted to integers.

This violates users' reasonable expectations that:
```python
s == pd.read_json(io.StringIO(s.to_json(orient='index')), typ='series', orient='index')
```

The behavior affects real-world usage where:
- Users serialize numerical data with specific dtypes (e.g., for consistency with C libraries or for memory-mapped arrays) and expect those dtypes to be preserved
- Users use string labels that happen to be numeric (e.g., IDs, postal codes) and expect them to remain strings

## Fix

The documentation should prominently show that true round-trip preservation requires disabling type inference:

```diff
 Encoding/decoding a Dataframe using ``'split'`` formatted JSON:

 >>> df.to_json(orient='split')
     '{{"columns":["col 1","col 2"],"index":["row 1","row 2"],"data":[["a","b"],["c","d"]]}}'
->>> pd.read_json(StringIO(_), orient='split')
+>>> pd.read_json(StringIO(_), orient='split', dtype=False, convert_axes=False)
       col 1 col 2
 row 1     a     b
 row 2     c     d
```

Alternatively, the API could provide a dedicated round-trip method:

```python
@classmethod
def from_json(cls, json_str, orient=None):
    """Read JSON produced by to_json() with full type preservation."""
    return read_json(io.StringIO(json_str), typ='series', orient=orient,
                     dtype=False, convert_axes=False)
```

Or add a `roundtrip=True` parameter to `read_json` that sets both `dtype=False` and `convert_axes=False`.