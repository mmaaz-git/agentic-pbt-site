# Bug Report: pandas.io.json - Data Loss in Column Names During JSON Round-Trip

**Target**: `pandas.io.json` (specifically `pd.read_json` and `df.to_json`)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When a DataFrame with all-numeric string column names is serialized to JSON and deserialized back, pandas silently converts string column names to integers, causing data loss when leading zeros are present (e.g., '00' becomes 0, '01234' becomes 1234).

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
from io import StringIO


@given(
    columns=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=5, unique=True),
    num_rows=st.integers(min_value=1, max_value=20)
)
@settings(max_examples=200)
def test_json_roundtrip_column_names(columns, num_rows):
    data = {col: list(range(num_rows)) for col in columns}
    df = pd.DataFrame(data)

    json_str = df.to_json(orient='split')
    result = pd.read_json(StringIO(json_str), orient='split')

    assert list(result.columns) == list(df.columns), f"Column names changed from {list(df.columns)} to {list(result.columns)}"

# Run the test
if __name__ == "__main__":
    test_json_roundtrip_column_names()
```

<details>

<summary>
**Failing input**: `columns=['0'], num_rows=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 22, in <module>
    test_json_roundtrip_column_names()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 7, in test_json_roundtrip_column_names
    columns=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=5, unique=True),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 18, in test_json_roundtrip_column_names
    assert list(result.columns) == list(df.columns), f"Column names changed from {list(df.columns)} to {list(result.columns)}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Column names changed from ['0'] to [0]
Falsifying example: test_json_roundtrip_column_names(
    columns=['0'],
    num_rows=1,
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
from io import StringIO

# Create a DataFrame with all-numeric string column names
df = pd.DataFrame({'00': [1, 2], '0': [3, 4]})
print("Original DataFrame:")
print(df)
print("\nOriginal column names:", list(df.columns))
print("Original column types:", [type(c) for c in df.columns])

# Convert to JSON
json_str = df.to_json(orient='split')
print("\nJSON representation:")
print(json_str)

# Read back from JSON
result = pd.read_json(StringIO(json_str), orient='split')
print("\nResulting DataFrame after round-trip:")
print(result)
print("\nResult column names:", list(result.columns))
print("Result column types:", [type(c) for c in result.columns])

# Check for data loss
print("\nData loss detected:")
print(f"  '00' became: {result.columns[0]} (type: {type(result.columns[0])})")
print(f"  '0' became: {result.columns[1]} (type: {type(result.columns[1])})")

# Show that both columns became the same value
if result.columns[0] == result.columns[1]:
    print(f"\nCRITICAL: Both columns '00' and '0' became the same value: {result.columns[0]}")
```

<details>

<summary>
Data loss demonstration: '00' and '0' both become integer 0
</summary>
```
Original DataFrame:
   00  0
0   1  3
1   2  4

Original column names: ['00', '0']
Original column types: [<class 'str'>, <class 'str'>]

JSON representation:
{"columns":["00","0"],"index":[0,1],"data":[[1,3],[2,4]]}

Resulting DataFrame after round-trip:
   0  0
0  1  3
1  2  4

Result column names: [0, 0]
Result column types: [<class 'int'>, <class 'int'>]

Data loss detected:
  '00' became: 0 (type: <class 'numpy.int64'>)
  '0' became: 0 (type: <class 'numpy.int64'>)

CRITICAL: Both columns '00' and '0' became the same value: 0
```
</details>

## Why This Is A Bug

This violates expected behavior for several critical reasons:

1. **Irreversible data loss**: Column names '00' and '0' both become integer 0, losing the distinction between them and the leading zeros. This is not just a type conversion but actual information loss.

2. **Silent corruption**: The conversion happens silently without any warning, potentially corrupting important data like ZIP codes (01234 → 1234), product codes, or other zero-padded identifiers.

3. **Inconsistent behavior**: When columns contain a mix of numeric and non-numeric strings (e.g., ['0', 'a']), they are preserved as strings. Only all-numeric column sets trigger the conversion.

4. **JSON format contradiction**: The JSON output correctly stores columns as strings (`"columns":["00","0"]`), but `read_json()` ignores this and converts them anyway.

5. **Documentation gap**: While the `convert_axes` parameter documentation mentions "Try to convert the axes to the proper dtypes", it doesn't warn about data loss scenarios or specify what "proper" means.

6. **Round-trip property violation**: A fundamental expectation of serialization/deserialization is that `deserialize(serialize(data)) == data`, which is violated here.

## Relevant Context

- **pandas version**: 2.3.2
- **Workaround available**: Setting `convert_axes=False` in `pd.read_json()` preserves the original column names
- **Default behavior**: `convert_axes` defaults to `True` for all orient values except 'table'
- **Real-world impact**: Common use cases include ZIP codes, product codes, account numbers, and any other identifiers that may be zero-padded
- **Code location**: The conversion happens in `/pandas/io/json/_json.py` in the `_convert_axes()` method (line 1189)

Documentation: https://pandas.pydata.org/docs/reference/api/pandas.read_json.html

## Proposed Fix

The issue can be fixed by modifying the `_convert_axes` method to avoid converting column names when it would cause data loss:

```diff
--- a/pandas/io/json/_json.py
+++ b/pandas/io/json/_json.py
@@ -1195,6 +1195,13 @@ class FrameParser(Parser):
         for axis_name in obj._AXIS_ORDERS:
             ax = obj._get_axis(axis_name)
             ser = Series(ax, dtype=ax.dtype, copy=False)
+
+            # Skip conversion for columns if it would cause data loss
+            # (e.g., '00' -> 0 loses the leading zero)
+            if axis_name == 'columns' and all(isinstance(x, str) for x in ax):
+                # Check if any values would lose information when converted
+                if any(str(pd.to_numeric(x, errors='coerce')) != x for x in ax if pd.api.types.is_number(pd.to_numeric(x, errors='coerce'))):
+                    continue
             new_ser, result = self._try_convert_data(
                 name=axis_name,
                 data=ser,
```

Alternatively, change the default value of `convert_axes` to `False` to prevent unexpected data loss, though this would be a breaking change.