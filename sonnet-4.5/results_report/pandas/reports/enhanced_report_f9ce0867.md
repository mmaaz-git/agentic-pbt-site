# Bug Report: pandas.io.sas._convert_datetimes Fractional Days Truncation

**Target**: `pandas.io.sas.sas7bdat._convert_datetimes`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_convert_datetimes` function silently truncates fractional day components when unit='d', causing data loss of up to 23 hours, 59 minutes, and 59 seconds, while its sister function `_parse_datetime` correctly preserves fractional days.

## Property-Based Test

```python
from hypothesis import given, strategies as st, example
import pandas as pd
from pandas.io.sas.sas7bdat import _parse_datetime, _convert_datetimes


@given(
    values=st.lists(
        st.floats(min_value=-1e8, max_value=1e8, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=20
    )
)
@example(values=[0.5])  # The failing example from the bug report
def test_convert_datetimes_consistency_with_parse_datetime(values):
    series = pd.Series(values)
    vectorized_result = _convert_datetimes(series, "d")

    for i, value in enumerate(values):
        scalar_result = _parse_datetime(value, "d")
        vectorized_value = vectorized_result.iloc[i]

        scalar_ts = pd.Timestamp(scalar_result)
        vectorized_ts = pd.Timestamp(vectorized_value)

        time_diff_ms = abs((scalar_ts - vectorized_ts).total_seconds() * 1000)

        assert time_diff_ms < 1, (
            f"Inconsistency at index {i} for value {value} with unit d: "
            f"_parse_datetime returned {scalar_result}, "
            f"_convert_datetimes returned {vectorized_value}, "
            f"difference: {time_diff_ms}ms"
        )

# Run the test
if __name__ == "__main__":
    test_convert_datetimes_consistency_with_parse_datetime()
```

<details>

<summary>
**Failing input**: `values=[0.5]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/24/hypo.py", line 36, in <module>
    test_convert_datetimes_consistency_with_parse_datetime()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/24/hypo.py", line 7, in test_convert_datetimes_consistency_with_parse_datetime
    values=st.lists(
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/24/hypo.py", line 27, in test_convert_datetimes_consistency_with_parse_datetime
    assert time_diff_ms < 1, (
           ^^^^^^^^^^^^^^^^
AssertionError: Inconsistency at index 0 for value 0.5 with unit d: _parse_datetime returned 1960-01-01 12:00:00, _convert_datetimes returned 1960-01-01 00:00:00, difference: 43200000.0ms
Falsifying explicit example: test_convert_datetimes_consistency_with_parse_datetime(
    values=[0.5],
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
from pandas.io.sas.sas7bdat import _parse_datetime, _convert_datetimes

# Test cases with fractional days
test_values = [0.25, 0.5, 0.75, 1.5, 2.33]

print("Demonstrating the bug: _convert_datetimes truncates fractional days\n")
print("="*70)

for value in test_values:
    # Test _parse_datetime (scalar function)
    parse_result = _parse_datetime(value, "d")

    # Test _convert_datetimes (vectorized function)
    series = pd.Series([value])
    convert_result = _convert_datetimes(series, "d").iloc[0]

    # Calculate the difference
    parse_ts = pd.Timestamp(parse_result)
    convert_ts = pd.Timestamp(convert_result)
    diff = parse_ts - convert_ts

    print(f"Input value: {value} days")
    print(f"  _parse_datetime result:    {parse_result}")
    print(f"  _convert_datetimes result: {convert_result}")
    print(f"  Difference (data lost):    {diff}")
    print(f"  Hours lost: {diff.total_seconds() / 3600:.2f}")
    print("-"*70)

print("\nSummary: All fractional day components are silently truncated!")
print("This causes significant data loss when reading SAS files with fractional days.")
```

<details>

<summary>
Demonstrating silent data loss across multiple fractional day values
</summary>
```
Demonstrating the bug: _convert_datetimes truncates fractional days

======================================================================
Input value: 0.25 days
  _parse_datetime result:    1960-01-01 06:00:00
  _convert_datetimes result: 1960-01-01 00:00:00
  Difference (data lost):    0 days 06:00:00
  Hours lost: 6.00
----------------------------------------------------------------------
Input value: 0.5 days
  _parse_datetime result:    1960-01-01 12:00:00
  _convert_datetimes result: 1960-01-01 00:00:00
  Difference (data lost):    0 days 12:00:00
  Hours lost: 12.00
----------------------------------------------------------------------
Input value: 0.75 days
  _parse_datetime result:    1960-01-01 18:00:00
  _convert_datetimes result: 1960-01-01 00:00:00
  Difference (data lost):    0 days 18:00:00
  Hours lost: 18.00
----------------------------------------------------------------------
Input value: 1.5 days
  _parse_datetime result:    1960-01-02 12:00:00
  _convert_datetimes result: 1960-01-02 00:00:00
  Difference (data lost):    0 days 12:00:00
  Hours lost: 12.00
----------------------------------------------------------------------
Input value: 2.33 days
  _parse_datetime result:    1960-01-03 07:55:12
  _convert_datetimes result: 1960-01-03 00:00:00
  Difference (data lost):    0 days 07:55:12
  Hours lost: 7.92
----------------------------------------------------------------------

Summary: All fractional day components are silently truncated!
This causes significant data loss when reading SAS files with fractional days.
```
</details>

## Why This Is A Bug

1. **Silent Data Loss**: The function truncates fractional day components without warning, losing up to 23:59:59 of time data. A value of 0.99 days (23 hours and 45.6 minutes) becomes 0 days.

2. **Inconsistent API Behavior**: Two functions in the same module designed for the same purpose behave differently:
   - `_parse_datetime(0.5, "d")` correctly returns `1960-01-01 12:00:00`
   - `_convert_datetimes([0.5], "d")` incorrectly returns `1960-01-01 00:00:00`

3. **Violates SAS Format Standards**: SAS date values are stored as floating-point numbers of days since January 1, 1960, and the format explicitly supports fractional days to represent time within a day.

4. **Contradicts Function Documentation**: The docstring states "Convert to Timestamp if possible" and mentions "SAS float64 lacks precision for more than ms resolution" but does not document that fractional days will be truncated to whole days.

5. **Asymmetric Implementation**: The same function correctly preserves fractional components when unit='s' (seconds), but truncates them when unit='d' (days), creating an unexpected inconsistency within the function itself.

## Relevant Context

The bug occurs at line 107 in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/io/sas/sas7bdat.py`:

```python
vals = np.array(sas_datetimes, dtype="M8[D]") + td
```

The `dtype="M8[D]"` specifies datetime64 with day precision, which can only store whole days. This is inconsistent with:
- Line 75 where `_parse_datetime` uses `timedelta(days=sas_datetime)` which preserves fractional days
- Lines 101-104 where the 's' (seconds) path uses millisecond precision via `cast_from_unit_vectorized`

Both functions are used internally by pandas' public `read_sas()` function when processing SAS7BDAT files. While the functions are internal (underscore-prefixed), they directly impact data integrity for users reading SAS files.

SAS documentation: https://documentation.sas.com/doc/en/pgmsascdc/9.4_3.4/lrcon/p1wj0povp3j7ccn1kp4ht82a2t1o.htm

## Proposed Fix

```diff
--- a/pandas/io/sas/sas7bdat.py
+++ b/pandas/io/sas/sas7bdat.py
@@ -104,8 +104,11 @@ def _convert_datetimes(sas_datetimes: pd.Series, unit: str) -> pd.Series:
         dt64ms = millis.view("M8[ms]") + td
         return pd.Series(dt64ms, index=sas_datetimes.index, copy=False)
     else:
-        vals = np.array(sas_datetimes, dtype="M8[D]") + td
-        return pd.Series(vals, dtype="M8[s]", index=sas_datetimes.index, copy=False)
+        millis = cast_from_unit_vectorized(
+            sas_datetimes._values, unit="D", out_unit="ms"
+        )
+        dt64ms = millis.view("M8[ms]") + td
+        return pd.Series(dt64ms, index=sas_datetimes.index, copy=False)


 class _Column:
```