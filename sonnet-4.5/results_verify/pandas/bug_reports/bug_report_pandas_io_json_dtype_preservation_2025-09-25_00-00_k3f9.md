# Bug Report: pandas.io.json dtype Preservation Failure

**Target**: `pandas.io.json.read_json` / `pandas.io.json.to_json`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Float columns with integer-like values (e.g., 0.0) are incorrectly converted to int64 during JSON round-trip, violating the documented round-trip property.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
from pandas.io.json import read_json, to_json
import io

@given(
    data=st.lists(
        st.fixed_dictionaries({
            'a': st.integers(min_value=-1000, max_value=1000),
            'b': st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        }),
        min_size=0,
        max_size=20
    )
)
@settings(max_examples=100)
def test_roundtrip_orient_records(data):
    df = pd.DataFrame(data)
    json_str = to_json(None, df, orient='records')
    result = read_json(io.StringIO(json_str), orient='records')
    pd.testing.assert_frame_equal(result, df)
```

**Failing input**: `[{'a': 0, 'b': 0.0}]`

## Reproducing the Bug

```python
import pandas as pd
from pandas.io.json import read_json, to_json
import io

df = pd.DataFrame([{'b': 0.0}])
print(f"Original dtype: {df['b'].dtype}")

json_str = to_json(None, df, orient='records')
result = read_json(io.StringIO(json_str), orient='records')
print(f"Result dtype: {result['b'].dtype}")

assert df['b'].dtype == result['b'].dtype
```

Output:
```
Original dtype: float64
Result dtype: int64
AssertionError
```

## Why This Is A Bug

The `read_json` docstring states: "Compatible JSON strings can be produced by `to_json()` with a corresponding orient value." This implies round-trip correctness should be preserved.

This bug affects all orient values except 'table':
- ❌ `orient='records'`
- ❌ `orient='split'`
- ❌ `orient='columns'`
- ❌ `orient='index'`
- ✅ `orient='table'` (preserves dtypes via schema)

The workaround is to use `read_json(..., dtype={'b': 'float64'})`, but this requires the user to manually track types, defeating the purpose of the round-trip property.

## Fix

The issue occurs in the type inference logic during JSON reading. When `read_json` encounters numeric values without explicit dtype information, it uses pandas' default type inference, which converts integer-like floats to integers.

**Proposed solution**: When using orient values that don't include schema metadata ('records', 'split', 'columns', 'index', 'values'), `read_json` should:
1. Default to float64 for all numeric columns (safer, preserves precision)
2. OR provide a parameter like `infer_dtype=False` to disable aggressive type inference

**Alternatively**: Document this limitation clearly in both `to_json` and `read_json` docstrings, and recommend using `orient='table'` for exact round-trips.