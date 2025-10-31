# Bug Report: pandas.io.json ujson Float Precision Loss

**Target**: `pandas.io.json.ujson_dumps` and `pandas.io.json.ujson_loads`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `ujson_dumps` function loses precision for floating-point numbers, particularly for large values and values with many significant digits. This violates the expected round-trip property that `ujson_loads(ujson_dumps(x)) == x`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.io.json import ujson_dumps, ujson_loads
import pandas as pd

@given(
    st.one_of(
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(),
        st.booleans(),
        st.none()
    )
)
def test_ujson_roundtrip(data):
    json_str = ujson_dumps(data)
    result = ujson_loads(json_str)
    assert result == data or (pd.isna(result) and pd.isna(data))
```

**Failing input**: `data=1.0000000000000002e+16`

## Reproducing the Bug

```python
from pandas.io.json import ujson_dumps, ujson_loads
import json

test_values = [
    1.0000000000000002e+16,
    1.23456789012345678,
]

print("ujson precision loss:")
for val in test_values:
    json_str = ujson_dumps(val)
    result = ujson_loads(json_str)
    print(f"Original: {val}")
    print(f"JSON:     {json_str}")
    print(f"Result:   {result}")
    print(f"Equal:    {val == result}")
    print()

print("\nStandard json module (for comparison):")
for val in test_values:
    json_str = json.dumps(val)
    result = json.loads(json_str)
    print(f"Original: {val}")
    print(f"JSON:     {json_str}")
    print(f"Result:   {result}")
    print(f"Equal:    {val == result}")
    print()
```

Output:
```
ujson precision loss:
Original: 1.0000000000000002e+16
JSON:     1e+16
Result:   1e+16
Equal:    False

Original: 1.2345678901234567
JSON:     1.2345678901
Result:   1.2345678901
Equal:    False

Standard json module (for comparison):
Original: 1.0000000000000002e+16
JSON:     1.0000000000000002e+16
Result:   1.0000000000000002e+16
Equal:    True

Original: 1.2345678901234567
JSON:     1.2345678901234567
Result:   1.2345678901234567
Equal:    True
```

## Why This Is A Bug

While ujson is known for prioritizing speed over precision, this behavior is not documented in the pandas API. Users calling `ujson_dumps` from pandas would reasonably expect it to preserve floating-point precision like the standard library `json.dumps` does.

This is particularly problematic because:
1. The precision loss is silent - no warning or error is raised
2. It affects both very large numbers and numbers with many decimal places
3. The standard library doesn't have this issue
4. The pandas documentation doesn't mention this limitation

## Fix

There are several potential fixes:

**Option 1**: Use the `double_precision` parameter in `ujson_dumps` (currently hardcoded to 10 in some places):

```diff
--- a/pandas/io/json/_json.py
+++ b/pandas/io/json/_json.py
@@ -263,7 +263,7 @@ class Writer(ABC):
         return ujson_dumps(
             self.obj_to_write,
             orient=self.orient,
-            double_precision=self.double_precision,
+            double_precision=max(self.double_precision, 17),  # IEEE 754 double precision
             ensure_ascii=self.ensure_ascii,
             date_unit=self.date_unit,
             iso_dates=iso_dates,
```

**Option 2**: Document the limitation clearly in the docstring:

```diff
--- a/pandas/io/json/_json.py
+++ b/pandas/io/json/_json.py
@@ -25,6 +25,10 @@ from pandas._libs.json import (
     ujson_dumps,
     ujson_loads,
 )
+# Note: ujson_dumps may lose precision for floating-point numbers.
+# For full precision, use the standard library json.dumps() instead.
```

**Option 3**: Add a parameter to allow users to choose precision vs. speed:

```python
def ujson_dumps(obj, precise=False, **kwargs):
    if precise:
        kwargs['double_precision'] = 17
    return _ujson_dumps(obj, **kwargs)
```

The recommended fix is Option 1, as it makes the default behavior safer without significantly impacting performance.