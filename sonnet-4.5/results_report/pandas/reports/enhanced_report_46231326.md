# Bug Report: pandas.io.json.read_json Converts Empty Strings to NaT During Round-Trip

**Target**: `pandas.io.json.read_json`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When serializing and deserializing a pandas Series containing empty strings through `to_json()`/`read_json()`, empty strings are silently converted to `NaT` (Not-a-Time) datetime values due to overly aggressive automatic date conversion, causing data corruption and violating the expected round-trip property.

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


if __name__ == "__main__":
    test_series_roundtrip_split()
```

<details>

<summary>
**Failing input**: `data=['']`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 17, in <module>
    test_series_roundtrip_split()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 7, in test_series_roundtrip_split
    @settings(max_examples=500)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 13, in test_series_roundtrip_split
    pd.testing.assert_series_equal(s, s_recovered, check_dtype=False)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 1091, in assert_series_equal
    _testing.assert_almost_equal(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        left._values,
        ^^^^^^^^^^^^^
    ...<5 lines>...
        index_values=left.index,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "pandas/_libs/testing.pyx", line 55, in pandas._libs.testing.assert_almost_equal
  File "pandas/_libs/testing.pyx", line 173, in pandas._libs.testing.assert_almost_equal
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 620, in raise_assert_detail
    raise AssertionError(msg)
AssertionError: Series are different

Series values are different (100.0 %)
[index]: [0]
[left]:  []
[right]: <DatetimeArray>
['NaT']
Length: 1, dtype: datetime64[ns]
At positional index 0, first diff:  != NaT
Falsifying example: test_series_roundtrip_split(
    data=[''],
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/_dtype.py:364
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/_config/config.py:138
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/_config/config.py:628
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/_config/config.py:659
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/_config/config.py:685
        (and 35 more with settings.verbosity >= verbose)
```
</details>

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

# Also test with convert_dates=False to verify the workaround
print("\n=== With convert_dates=False ===")
s_recovered_no_convert = pd.read_json(StringIO(json_str), typ='series', orient='split', convert_dates=False)
print(f"Recovered Series (no convert):\n{s_recovered_no_convert}")
print(f"Recovered dtype (no convert): {s_recovered_no_convert.dtype}")
print(f"Recovered value (no convert): {s_recovered_no_convert.iloc[0]!r}")
```

<details>

<summary>
Empty string '' becomes NaT after JSON round-trip
</summary>
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

=== With convert_dates=False ===
Recovered Series (no convert):
0
dtype: object
Recovered dtype (no convert): object
Recovered value (no convert): ''
```
</details>

## Why This Is A Bug

This behavior violates fundamental expectations and causes data corruption:

1. **Empty strings are not dates**: An empty string ('') is not a valid representation of any date or time in any standard format (ISO 8601, RFC 3339, etc.). Converting it to NaT is semantically incorrect.

2. **Violates round-trip property**: A core expectation for JSON serialization is that `deserialize(serialize(x))` should equal `x` for supported types. This bug breaks that contract - the data that goes in is not what comes out.

3. **Silent data corruption**: The conversion happens without any warning or error. Users' data is silently modified, changing both the data type (object → datetime64[ns]) and the value ('' → NaT).

4. **Inconsistent with explicit conversion**: When explicitly converting with `pd.to_datetime('')`, pandas raises an error by default. Only with `errors='coerce'` does it return NaT. The JSON reader applies this coercion implicitly without user consent.

5. **Different semantic meaning**: An empty string and NaT have different meanings:
   - Empty string: "There is a value, and it is empty"
   - NaT: "There is no time value / missing temporal data"

   This distinction matters for data integrity and downstream processing.

## Relevant Context

The bug occurs because `read_json()` has `convert_dates=True` by default, which attempts to parse string values as dates. The date conversion logic appears to treat empty strings as parseable dates, likely because:

- The conversion attempts to coerce various string formats to dates
- Empty strings are being caught in this coercion logic
- There's no explicit check to exclude empty strings from date parsing

Documentation reference: [pandas.read_json](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_json.html)

The workaround of setting `convert_dates=False` preserves empty strings correctly, confirming that the automatic date conversion is the root cause. However, this disables all automatic date conversion, which may not be desirable if the data contains actual date columns.

## Proposed Fix

The date conversion logic should explicitly check for and skip empty strings before attempting to parse them as dates. Here's a high-level approach:

```diff
--- a/pandas/io/json/_json.py
+++ b/pandas/io/json/_json.py
@@ -[date_conversion_method]
     def _try_convert_to_date(self, data):
         if self.convert_dates:
+            # Don't attempt to convert empty strings to dates
+            if isinstance(data, str) and data == '':
+                return data, False
+            # For array-like data, check for empty strings
+            if hasattr(data, '__iter__') and not isinstance(data, str):
+                if any(isinstance(x, str) and x == '' for x in data):
+                    # Don't convert if any empty strings present
+                    return data, False

             # Existing date conversion logic
             try:
                 converted = pd.to_datetime(data, errors='coerce')
```

A more robust fix would involve reviewing the entire date inference logic to ensure it only converts strings that actually look like dates (contain date-like patterns such as digits, hyphens, slashes, colons, etc.) rather than blindly attempting to parse all strings.