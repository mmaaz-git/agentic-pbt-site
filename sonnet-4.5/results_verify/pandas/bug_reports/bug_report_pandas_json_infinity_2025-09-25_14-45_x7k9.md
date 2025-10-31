# Bug Report: pandas to_json/read_json Converts Max Float64 to Infinity

**Target**: `pandas.DataFrame.to_json` / `pandas.read_json`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Round-trip serialization through JSON (to_json → read_json) silently corrupts the maximum float64 value by converting it to infinity. This causes data loss for valid finite floating-point values near the upper limit of float64.

## Property-Based Test

```python
from hypothesis import given, settings
from hypothesis.extra.pandas import column, data_frames, range_indexes
import pandas as pd
import numpy as np
import io

@given(
    df=data_frames([
        column('a', dtype=int),
        column('b', dtype=float),
    ], index=range_indexes(min_size=0, max_size=50))
)
@settings(max_examples=100)
def test_to_json_read_json_roundtrip(df):
    json_buffer = io.StringIO()
    df.to_json(json_buffer, orient='records')
    json_buffer.seek(0)
    result = pd.read_json(json_buffer, orient='records')

    if len(df) > 0:
        pd.testing.assert_frame_equal(
            df.reset_index(drop=True),
            result.reset_index(drop=True),
            check_dtype=False
        )
```

**Failing input**: DataFrame containing `np.finfo(np.float64).max` (≈1.797693e+308)

```
       a              b
    0  0  1.797693e+308
```

## Reproducing the Bug

```python
import pandas as pd
import numpy as np
import io

df = pd.DataFrame({'value': [np.finfo(np.float64).max]})

print("Original value:", df['value'].iloc[0])
print("Is finite?", np.isfinite(df['value'].iloc[0]))

json_buffer = io.StringIO()
df.to_json(json_buffer, orient='records')
json_buffer.seek(0)

result = pd.read_json(json_buffer, orient='records')

print("\nAfter JSON round-trip:", result['value'].iloc[0])
print("Is finite?", np.isfinite(result['value'].iloc[0]))
```

**Output:**
```
Original value: 1.7976931348623157e+308
Is finite? True

After JSON round-trip: inf
Is finite? False
```

## Why This Is A Bug

This is a serious data corruption bug:

1. **Silent data loss**: Valid finite float64 values are silently converted to infinity without warning
2. **Violates round-trip property**: Serialization should preserve data, but `df.to_json()` → `pd.read_json()` corrupts values
3. **Affects real data**: Scientific computing, physics simulations, and financial data often use values near float64 limits
4. **Unexpected behavior**: Users expect JSON serialization to preserve numeric precision within the limits of the format

The bug occurs because:
1. `to_json()` outputs max float64 as `1.797693135e+308` (truncated precision)
2. The JSON parser in `read_json()` attempts to parse this string
3. The parsed value rounds up slightly, exceeding max float64
4. Values exceeding max float64 are represented as infinity

## Fix

```diff
--- a/pandas/io/json/_json.py
+++ b/pandas/io/json/_json.py
@@ -somewhere
 def to_json(...):
-    # Current: uses default JSON serialization which truncates precision
+    # Use higher precision for float serialization to preserve exact values
+    # Option 1: Use repr() for full precision
+    # Option 2: Use a custom JSON encoder that preserves float64 exactly
+    # Option 3: Document this limitation and recommend using pickle for exact precision
```

A proper fix would require one of:

1. **Increase JSON precision**: Use `double_precision=17` or higher in the JSON encoder to ensure all float64 values roundtrip exactly
2. **Clamp on read**: When parsing JSON, clamp values to `[-max_float64, max_float64]` instead of allowing overflow to infinity
3. **Use alternative encoding**: For values near limits, encode as a string with full precision and a type marker

**Workaround**: For now, users can avoid this by:
```python
df.to_json(path, orient='records', double_precision=17)
```

Or use pickle for exact serialization:
```python
df.to_pickle(path)
```