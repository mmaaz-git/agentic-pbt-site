# Bug Report: pandas.io.json Empty String Becomes NaT

**Target**: `pandas.io.json.read_json` (automatic date conversion)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When round-tripping a Series containing empty strings through `to_json`/`read_json`, the empty strings are incorrectly converted to `NaT` (Not-a-Time) datetime values. This happens due to overly aggressive automatic date conversion.

## Property-Based Test

```python
from io import StringIO
import pandas as pd
from hypothesis import given, settings, strategies as st


@given(st.lists(st.one_of(st.integers(), st.floats(allow_nan=False, allow_infinity=False), st.text()), min_size=1, max_size=20))
@settings(max_examples=500)
def test_series_roundtrip_split(data):
    s = pd.Series(data)
    json_str = s.to_json(orient='split')
    s_recovered = pd.read_json(StringIO(json_str), typ='series', orient='split')

    pd.testing.assert_series_equal(s, s_recovered, check_dtype=False)
```

**Failing input**: `data=['']`

## Reproducing the Bug

```python
from io import StringIO
import pandas as pd

s = pd.Series([''])
print(f"Original Series:\n{s}")
print(f"Original dtype: {s.dtype}")
print(f"Original value: {s.iloc[0]!r}")

json_str = s.to_json(orient='split')
print(f"\nJSON: {json_str}")

s_recovered = pd.read_json(StringIO(json_str), typ='series', orient='split')
print(f"\nRecovered Series:\n{s_recovered}")
print(f"Recovered dtype: {s_recovered.dtype}")
print(f"Recovered value: {s_recovered.iloc[0]!r}")
```

Output:
```
Original Series:
0
dtype: object
Original dtype: object
Original value: ''

JSON: {"name":null,"index":[0],"data":[""]}

Recovered Series:
0   NaT
dtype: datetime64[ns]
Recovered dtype: datetime64[ns]
Recovered value: NaT
```

## Why This Is A Bug

An empty string is not a valid date representation and should not be converted to `NaT`. The automatic date conversion in `read_json` (enabled by default with `convert_dates=True`) is too aggressive and incorrectly treats empty strings as dates.

This violates the round-trip property: `read_json(series.to_json()) != series` for series containing empty strings.

The bug can be worked around by setting `convert_dates=False`, which preserves empty strings correctly.

## Fix

```diff
--- a/pandas/io/json/_json.py
+++ b/pandas/io/json/_json.py
@@ -1321,6 +1321,10 @@ class Parser:
         if new_data.dtype == "object":
             try:
                 new_data = data.astype("int64")
+            except (TypeError, ValueError):
+                # Don't convert empty strings to dates
+                if (data == '').any():
+                    return data, False
             except OverflowError:
                 return data, False
             except (TypeError, ValueError):
```

Alternatively, the date conversion logic should explicitly check that strings are non-empty before attempting to parse them as dates.