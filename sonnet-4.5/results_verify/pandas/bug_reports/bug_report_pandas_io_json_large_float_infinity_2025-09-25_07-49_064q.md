# Bug Report: pandas.io.json Large Float Values Become Infinity

**Target**: `pandas.io.json` (specifically `to_json` and `read_json` with `orient='table'`)
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When round-tripping very large but finite float values through JSON with `orient='table'`, they are converted to infinity. The value `1.7976931345e+308` (which is less than `sys.float_info.max`) becomes `inf` after serialization and deserialization.

## Property-Based Test

```python
from hypothesis import given, settings
from hypothesis.extra.pandas import column, data_frames, range_indexes
import pandas as pd
import tempfile
import os


@given(
    data_frames(
        columns=[
            column("int_col", dtype=int),
            column("float_col", dtype=float),
            column("str_col", dtype=str),
        ],
        index=range_indexes(min_size=0, max_size=100),
    )
)
@settings(max_examples=100)
def test_json_round_trip_orient_table(df):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        temp_path = f.name

    try:
        df.to_json(temp_path, orient="table")
        result = pd.read_json(temp_path, orient="table")
        pd.testing.assert_frame_equal(df, result)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
```

**Failing input**: DataFrame with a float value of `1.7976931345e+308`

## Reproducing the Bug

```python
import pandas as pd
import tempfile
import sys

large_float = 1.7976931345e+308
print(f"sys.float_info.max: {sys.float_info.max}")
print(f"Test value: {large_float}")
print(f"Test value < max: {large_float < sys.float_info.max}")

df = pd.DataFrame({"value": [large_float]})
print(f"\nOriginal: {df['value'].iloc[0]}")

with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
    temp_path = f.name

df.to_json(temp_path, orient="table")
result = pd.read_json(temp_path, orient="table")

print(f"After round-trip: {result['value'].iloc[0]}")
print(f"Values match: {df['value'].iloc[0] == result['value'].iloc[0]}")
```

Output:
```
sys.float_info.max: 1.7976931348623157e+308
Test value: 1.7976931345e+308
Test value < max: True

Original: 1.7976931345e+308
After round-trip: inf
Values match: False
```

## Why This Is A Bug

This is a **silent data corruption bug**. Valid finite floating-point values are being converted to infinity during JSON serialization/deserialization. This is particularly serious because:

1. **Silent corruption**: No error or warning is raised
2. **Mathematical incorrectness**: Finite values become infinite
3. **Affects scientific computing**: Large numbers near float max are legitimate in scientific/engineering applications
4. **Breaks round-trip property**: Data is not preserved

The issue appears to be related to precision loss in the JSON representation. When `1.7976931345e+308` is written to JSON, it becomes `1.797693135e+308` (with reduced precision), and when this is read back, it overflows to infinity.

## Fix

The issue is likely in the JSON serialization step where `double_precision` parameter controls the number of decimal places. The default `double_precision=10` may be insufficient for very large floats.

Potential fixes:

```diff
--- a/pandas/io/json/_json.py
+++ b/pandas/io/json/_json.py
@@ -somewhere
-    def to_json(..., double_precision: int = 10, ...):
+    def to_json(..., double_precision: int = 17, ...):
```

Increasing `double_precision` from 10 to 17 (the maximum precision for IEEE 754 double-precision floats) would ensure that all finite float values can be accurately represented in JSON without overflow.

Alternatively, add validation to detect when serialization would cause overflow:

```python
import numpy as np

def validate_json_serialization(df, precision=10):
    for col in df.select_dtypes(include=[np.float64]).columns:
        max_val = df[col].abs().max()
        if max_val > 1e308 / (10 ** (308 - precision)):
            raise ValueError(
                f"Column '{col}' contains values too large for double_precision={precision}. "
                f"Use double_precision=17 or higher."
            )
```