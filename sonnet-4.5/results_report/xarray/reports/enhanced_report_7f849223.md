# Bug Report: xarray.coding.times.CFDatetimeCoder Round-Trip Precision Loss

**Target**: `xarray.coding.times.CFDatetimeCoder`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

CFDatetimeCoder violates the documented contract that `coder.decode(coder.encode(variable)) == variable` by losing nanosecond-level precision when encoding datetime values with coarse time units like "days since".

## Property-Based Test

```python
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings
from xarray.core.variable import Variable
from xarray.coding.times import CFDatetimeCoder


@given(
    st.lists(
        st.datetimes(
            min_value=pd.Timestamp("2000-01-01"),
            max_value=pd.Timestamp("2050-12-31"),
        ),
        min_size=1,
        max_size=20,
    )
)
@settings(max_examples=100)
def test_datetime_coder_round_trip(datetime_list):
    datetime_arr = np.array(datetime_list, dtype="datetime64[ns]")

    encoding = {"units": "days since 2000-01-01", "calendar": "proleptic_gregorian"}
    original_var = Variable(("time",), datetime_arr, encoding=encoding)

    coder = CFDatetimeCoder()

    encoded_var = coder.encode(original_var)
    decoded_var = coder.decode(encoded_var)

    np.testing.assert_array_equal(original_var.data, decoded_var.data)
    assert original_var.dims == decoded_var.dims

if __name__ == "__main__":
    test_datetime_coder_round_trip()
```

<details>

<summary>
**Failing input**: `datetime_list=[datetime.datetime(2000, 1, 1, 0, 0, 0, 2003)]`
</summary>
```
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'hours' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'seconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
/home/npc/pbt/agentic-pbt/worker_/60/hypo.py:27: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 34, in <module>
    test_datetime_coder_round_trip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 9, in test_datetime_coder_round_trip
    st.lists(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 30, in test_datetime_coder_round_trip
    np.testing.assert_array_equal(original_var.data, decoded_var.data)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py", line 1051, in assert_array_equal
    assert_array_compare(operator.__eq__, actual, desired, err_msg=err_msg,
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                         verbose=verbose, header='Arrays are not equal',
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                         strict=strict)
                         ^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py", line 916, in assert_array_compare
    raise AssertionError(msg)
AssertionError:
Arrays are not equal

Mismatched elements: 1 / 1 (100%)
Max absolute difference among violations: 1
 ACTUAL: array(['2000-01-01T00:00:00.002003000'], dtype='datetime64[ns]')
 DESIRED: array(['2000-01-01T00:00:00.002002999'], dtype='datetime64[ns]')
Falsifying example: test_datetime_coder_round_trip(
    datetime_list=[datetime.datetime(2000, 1, 1, 0, 0, 0, 2003)],
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/_dtype.py:339
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py:862
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py:870
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py:871
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py:876
        (and 2 more with settings.verbosity >= verbose)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from datetime import datetime
from xarray.core.variable import Variable
from xarray.coding.times import CFDatetimeCoder

dt = datetime(2003, 1, 1, 0, 0, 0, 1)
datetime_arr = np.array([dt], dtype="datetime64[ns]")

encoding = {"units": "days since 2000-01-01", "calendar": "proleptic_gregorian"}
original_var = Variable(("time",), datetime_arr, encoding=encoding)

coder = CFDatetimeCoder()
encoded_var = coder.encode(original_var)
decoded_var = coder.decode(encoded_var)

print(f"Original: {original_var.data[0]} ({original_var.data.view('int64')[0]} ns)")
print(f"Decoded:  {decoded_var.data[0]} ({decoded_var.data.view('int64')[0]} ns)")
print(f"Lost precision: {original_var.data.view('int64')[0] - decoded_var.data.view('int64')[0]} ns")

assert np.array_equal(original_var.data, decoded_var.data), "Round-trip failed!"
```

<details>

<summary>
AssertionError: Round-trip failed!
</summary>
```
/home/npc/pbt/agentic-pbt/worker_/60/repo.py:13: UserWarning: Times can't be serialized faithfully to int64 with requested units 'days since 2000-01-01'. Resolution of 'microseconds' needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.
  encoded_var = coder.encode(original_var)
Original: 2003-01-01T00:00:00.000001000 (1041379200000001000 ns)
Decoded:  2003-01-01T00:00:00.000000976 (1041379200000000976 ns)
Lost precision: 24 ns
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/60/repo.py", line 20, in <module>
    assert np.array_equal(original_var.data, decoded_var.data), "Round-trip failed!"
           ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Round-trip failed!
```
</details>

## Why This Is A Bug

The `VariableCoder` base class in xarray explicitly documents a contract at `/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/xarray/coding/common.py:29-30`:

> "Subclasses should implement encode() and decode(), which should satisfy the identity `coder.decode(coder.encode(variable)) == variable`."

CFDatetimeCoder, as a subclass of VariableCoder, violates this mandatory contract. When encoding datetime values with microsecond precision using coarse units like "days since", the coder:

1. **Detects the precision loss**: The code calculates that "microseconds" resolution is needed (line 1086)
2. **Warns about it**: Emits a UserWarning about the precision loss (lines 1093-1098)
3. **Proceeds anyway**: Continues with lossy float64 encoding, corrupting the data by 24 nanoseconds

The issue occurs because float64 cannot accurately represent microsecond-level time differences when the base unit is "days". The limited precision of float64 (53 bits mantissa) causes rounding errors when converting between days (as float) and nanoseconds (as int64).

The code already has logic to handle this correctly - it can automatically adjust units when `allow_units_modification=True` and dtype is integer (lines 1099-1109), but this is not the default behavior for the general case.

## Relevant Context

The bug affects:
- Scientific data with microsecond/nanosecond precision (e.g., satellite observations, high-frequency sensors)
- Financial systems tracking microsecond-level timestamps
- Any workflow that saves and reloads xarray datasets expecting data fidelity

Workarounds exist:
- Use finer units: `"microseconds since 2000-01-01"` instead of `"days since 2000-01-01"`
- Explicitly set dtype to integer: `encoding={'dtype': np.int64}`
- Allow units modification in encoding settings

CF Conventions context: While CF conventions define time encoding standards, they don't mandate perfect round-trip precision. However, xarray's own documented contract does.

Relevant code locations:
- Contract definition: `xarray/coding/common.py:29-30`
- Bug location: `xarray/coding/times.py:1044-1166` (function `_eagerly_encode_cf_datetime`)
- Precision detection: `xarray/coding/times.py:1086-1087`

## Proposed Fix

```diff
diff --git a/xarray/coding/times.py b/xarray/coding/times.py
index abc123..def456 100644
--- a/xarray/coding/times.py
+++ b/xarray/coding/times.py
@@ -1089,11 +1089,18 @@ def _eagerly_encode_cf_datetime(
         floor_division = np.issubdtype(dtype, np.integer) or dtype is None
         if time_delta > needed_time_delta:
             floor_division = False
-            if dtype is None:
+            if dtype is None and not allow_units_modification:
                 emit_user_level_warning(
                     f"Times can't be serialized faithfully to int64 with requested units {units!r}. "
                     f"Resolution of {needed_units!r} needed. Serializing times to floating point instead. "
                     f"Set encoding['dtype'] to integer dtype to serialize to int64. "
                     f"Set encoding['dtype'] to floating point dtype to silence this warning."
                 )
+            elif dtype is None and allow_units_modification:
+                # Auto-adjust units to preserve precision when dtype is not specified
+                new_units = f"{needed_units} since {format_timestamp(ref_date)}"
+                units = new_units
+                time_delta = needed_time_delta
+                floor_division = True
             elif np.issubdtype(dtype, np.integer) and allow_units_modification:
```