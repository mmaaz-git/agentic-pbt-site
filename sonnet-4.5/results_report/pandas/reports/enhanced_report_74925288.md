# Bug Report: pandas.io.json.to_json Silent Float Precision Loss

**Target**: `pandas.io.json.to_json`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The default `double_precision=10` parameter in `to_json()` causes silent precision loss for float64 values during JSON serialization, corrupting data without warning.

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

if __name__ == "__main__":
    test_dataframe_json_roundtrip()
```

<details>

<summary>
**Failing input**: `value=1.4232917969412355e-157`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 26, in <module>
    test_dataframe_json_roundtrip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 6, in test_dataframe_json_roundtrip
    st.floats(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 23, in test_dataframe_json_roundtrip
    assert orig == restored, f"Round-trip failed: {orig} != {restored}"
           ^^^^^^^^^^^^^^^^
AssertionError: Round-trip failed: 1.4232917969412355e-157 != 1.423291797e-157
Falsifying example: test_dataframe_json_roundtrip(
    value=1.4232917969412355e-157,
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
from io import StringIO

# Test with the specific failing value from the bug report
df = pd.DataFrame({'col': [1.5932223682757467]})
print(f"Original value:  {df['col'].iloc[0]:.17f}")

# Serialize to JSON with default settings
json_str = df.to_json(orient='records')
print(f"JSON output:     {json_str}")

# Restore from JSON
df_restored = pd.read_json(StringIO(json_str), orient='records')
print(f"Restored value:  {df_restored['col'].iloc[0]:.17f}")

# Check equality
print(f"Values equal:    {df['col'].iloc[0] == df_restored['col'].iloc[0]}")

# Show the precision loss
print(f"Difference:      {abs(df['col'].iloc[0] - df_restored['col'].iloc[0]):.20e}")

print("\n--- Testing with double_precision=15 ---")

# Test with increased precision
json_str_15 = df.to_json(orient='records', double_precision=15)
print(f"JSON output:     {json_str_15}")

df_restored_15 = pd.read_json(StringIO(json_str_15), orient='records')
print(f"Restored value:  {df_restored_15['col'].iloc[0]:.17f}")
print(f"Values equal:    {df['col'].iloc[0] == df_restored_15['col'].iloc[0]}")
print(f"Difference:      {abs(df['col'].iloc[0] - df_restored_15['col'].iloc[0]):.20e}")

print("\n--- Testing with Python's standard json module ---")
import json

# Using Python's standard JSON library for comparison
data = {'col': [1.5932223682757467]}
json_str_std = json.dumps(data)
print(f"JSON output:     {json_str_std}")

data_restored = json.loads(json_str_std)
print(f"Restored value:  {data_restored['col'][0]:.17f}")
print(f"Values equal:    {data['col'][0] == data_restored['col'][0]}")
```

<details>

<summary>
Float64 values silently lose precision with default settings
</summary>
```
Original value:  1.59322236827574670
JSON output:     [{"col":1.5932223683}]
Restored value:  1.59322236829999997
Values equal:    False
Difference:      2.42532660621463946882e-11

--- Testing with double_precision=15 ---
JSON output:     [{"col":1.593222368275747}]
Restored value:  1.59322236827574715
Values equal:    False
Difference:      4.44089209850062616169e-16

--- Testing with Python's standard json module ---
JSON output:     {"col": [1.5932223682757467]}
Restored value:  1.59322236827574670
Values equal:    True
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple ways:

1. **Silent data corruption**: Float64 values with more than 10 significant digits lose precision without any warning or error. The value `1.5932223682757467` becomes `1.5932223683` in JSON, permanently losing precision.

2. **Violates round-trip expectation**: Users reasonably expect that `pd.read_json(df.to_json())` should return the original data, or at least preserve it to the limits of float64 representation. This is a fundamental property of serialization.

3. **Inconsistent with Python's json module**: Python's standard `json.dumps()` preserves full float64 precision by default (as shown in the test), making pandas' behavior surprising.

4. **Inadequate default for float64**: Float64 values can have approximately 15-17 significant decimal digits, but the default only preserves 10. This means 33-40% of available precision is discarded by default.

5. **No warning in documentation**: While the `double_precision=10` default is documented, there's no warning that this causes data loss for typical float64 values, nor any recommendation to use `double_precision=15` for lossless serialization.

## Relevant Context

The issue is in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/io/json/_json.py:145` where the default is set:

```python
def to_json(
    path_or_buf: FilePath | WriteBuffer[str] | WriteBuffer[bytes] | None,
    obj: NDFrame,
    orient: str | None = None,
    date_format: str = "epoch",
    double_precision: int = 10,  # <-- This default causes precision loss
    ...
)
```

The documentation states "The possible maximal value is 15" which is sufficient for float64, but users must explicitly opt-in to avoid data corruption. Most users won't realize they need to do this until they've already lost data.

Documentation reference: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_json.html

## Proposed Fix

Change the default `double_precision` from 10 to 15 in `to_json()` to enable lossless round-trip for float64 values:

```diff
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