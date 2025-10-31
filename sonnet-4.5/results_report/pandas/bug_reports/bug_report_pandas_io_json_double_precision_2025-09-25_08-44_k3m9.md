# Bug Report: pandas.io.json Default Float Precision Loss

**Target**: `pandas.io.json.to_json` / `pandas.io.json.ujson_dumps`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The default `double_precision=10` parameter in `to_json()` and `ujson_dumps()` causes silent precision loss for float64 values during serialization, violating the fundamental round-trip property that users expect: `read_json(df.to_json()) ≈ df`.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import pandas as pd
from io import StringIO

@given(
    st.floats(
        allow_nan=False,
        allow_infinity=False,
        min_value=-1e10,
        max_value=1e10,
        width=64
    )
)
@settings(max_examples=500)
def test_dataframe_json_roundtrip(value):
    df = pd.DataFrame({'col': [value]})
    json_str = df.to_json(orient='records')
    df_restored = pd.read_json(StringIO(json_str), orient='records')

    orig = df['col'].iloc[0]
    restored = df_restored['col'].iloc[0]

    assert orig == restored, f"Round-trip failed: {orig} != {restored}"
```

**Failing input**: `value=1.5932223682757467`

## Reproducing the Bug

```python
import pandas as pd
from io import StringIO

df = pd.DataFrame({'values': [1.5932223682757467, 0.0013606744423365084]})
print(f"Original:  {df['values'].iloc[0]:.17f}")

json_str = df.to_json(orient='records')
print(f"JSON:      {json_str}")

df_restored = pd.read_json(StringIO(json_str), orient='records')
print(f"Restored:  {df_restored['values'].iloc[0]:.17f}")
print(f"Match:     {df['values'].iloc[0] == df_restored['values'].iloc[0]}")
```

**Output:**
```
Original:  1.59322236827574670
JSON:      [{"values":1.5932223683}]
Restored:  1.59322236830000000
Match:     False
```

The value `1.5932223682757467` becomes `1.5932223683` after round-trip through JSON serialization.

## Why This Is A Bug

1. **Silent data corruption**: Float values with more than 10 significant digits lose precision without warning
2. **Violates round-trip expectation**: The fundamental contract of serialization is that `deserialize(serialize(x)) ≈ x`
3. **Inadequate documentation**: While the default `double_precision=10` is documented, the docstring does not warn users that this causes precision loss for typical float64 values
4. **Poor default choice**: Float64 values can have ~15-17 significant decimal digits, but the default only preserves 10
5. **Inconsistent with stdlib**: Python's standard `json` module preserves full float precision by default

The docstring states "The possible maximal value is 15", which is sufficient for float64, but users must explicitly opt-in to avoid data corruption.

## Fix

Change the default `double_precision` from 10 to 15 in `to_json()` to enable lossless round-trip for float64 values:

```diff
diff --git a/pandas/io/json/_json.py b/pandas/io/json/_json.py
index 1234567..abcdef0 100644
--- a/pandas/io/json/_json.py
+++ b/pandas/io/json/_json.py
@@ -142,7 +142,7 @@ def to_json(
     obj: NDFrame,
     orient: str | None = None,
     date_format: str = "epoch",
-    double_precision: int = 10,
+    double_precision: int = 15,
     force_ascii: bool = True,
     date_unit: str = "ms",
     default_handler: Callable[[Any], JSONSerializable] | None = None,
```

Alternatively, if backward compatibility is a concern, add a prominent warning to the docstring explaining that the default `double_precision=10` may cause precision loss and users should use `double_precision=15` for lossless float64 serialization.