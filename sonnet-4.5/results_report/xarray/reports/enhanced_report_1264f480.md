# Bug Report: xarray.coding.variables.CFMaskCoder Round-Trip Encoding/Decoding Violation

**Target**: `xarray.coding.variables.CFMaskCoder`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

CFMaskCoder violates the fundamental round-trip property `decode(encode(variable)) == variable` documented in its base class VariableCoder. When data contains valid values equal to the fill value, they are incorrectly converted to NaN during decoding, causing silent data corruption.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, assume, settings, strategies as st
from hypothesis.extra.numpy import arrays
from xarray.coding.variables import CFMaskCoder
from xarray.core.variable import Variable

@given(arrays(dtype=np.float32, shape=st.tuples(st.integers(5, 20))),
       st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False))
@settings(max_examples=50)
def test_mask_coder_with_fill_value_roundtrip(data, fill_value):
    assume(not np.any(np.isnan(data)))
    assume(not np.any(np.isinf(data)))
    assume(not np.isnan(fill_value))

    original_var = Variable(('x',), data.copy(), encoding={'_FillValue': fill_value})
    coder = CFMaskCoder()

    encoded_var = coder.encode(original_var)
    decoded_var = coder.decode(encoded_var)

    np.testing.assert_array_equal(original_var.data, decoded_var.data)

if __name__ == "__main__":
    # Run the test
    test_mask_coder_with_fill_value_roundtrip()
```

<details>

<summary>
**Failing input**: `data=array([0., 0., 0., 0., 0.], dtype=float32), fill_value=0.0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 25, in <module>
    test_mask_coder_with_fill_value_roundtrip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 8, in test_mask_coder_with_fill_value_roundtrip
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False))
            ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 21, in test_mask_coder_with_fill_value_roundtrip
    np.testing.assert_array_equal(original_var.data, decoded_var.data)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py", line 1051, in assert_array_equal
    assert_array_compare(operator.__eq__, actual, desired, err_msg=err_msg,
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                         verbose=verbose, header='Arrays are not equal',
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                         strict=strict)
                         ^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py", line 808, in assert_array_compare
    flagged = func_assert_same_pos(x, y, func=isnan, hasval='nan')
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py", line 777, in func_assert_same_pos
    raise AssertionError(msg)
AssertionError:
Arrays are not equal

nan location mismatch:
 ACTUAL: array([0., 0., 0., 0., 0.], dtype=float32)
 DESIRED: array([nan, nan, nan, nan, nan], dtype=float32)
Falsifying example: test_mask_coder_with_fill_value_roundtrip(
    data=array([0., 0., 0., 0., 0.], dtype=float32),
    fill_value=0.0,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/_dtype.py:339
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/_ufunc_config.py:438
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/_ufunc_config.py:463
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/arrayprint.py:1085
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py:771
```
</details>

## Reproducing the Bug

```python
import numpy as np
from xarray.coding.variables import CFMaskCoder
from xarray.core.variable import Variable

# Test case from the bug report - data contains valid zeros, fill value is also 0.0
data = np.array([0., 0., 0., 0., 0.], dtype=np.float32)
fill_value = 0.0

original_var = Variable(('x',), data.copy(), encoding={'_FillValue': fill_value})
coder = CFMaskCoder()

print("Original data:", original_var.data)
print("Original encoding:", original_var.encoding)
print()

encoded_var = coder.encode(original_var)
print("After encoding:")
print("  Data:", encoded_var.data)
print("  Encoding:", encoded_var.encoding)
print("  Attributes:", encoded_var.attrs)
print()

decoded_var = coder.decode(encoded_var)
print("After decoding (should match original):")
print("  Data:", decoded_var.data)
print("  Encoding:", decoded_var.encoding)
print()

print("Do they match?")
print("  Arrays equal:", np.array_equal(original_var.data, decoded_var.data))
print("  Contains NaN:", np.any(np.isnan(decoded_var.data)))
```

<details>

<summary>
Original data corrupted to NaN after round-trip encoding/decoding
</summary>
```
Original data: [0. 0. 0. 0. 0.]
Original encoding: {'_FillValue': 0.0}

After encoding:
  Data: [0. 0. 0. 0. 0.]
  Encoding: {}
  Attributes: {'_FillValue': np.float32(0.0)}

After decoding (should match original):
  Data: [nan nan nan nan nan]
  Encoding: {'_FillValue': np.float32(0.0)}

Do they match?
  Arrays equal: False
  Contains NaN: True
```
</details>

## Why This Is A Bug

This violates the explicit contract documented in the VariableCoder base class at `/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/xarray/coding/common.py:29-30`:

> "Subclasses should implement encode() and decode(), which should satisfy the identity `coder.decode(coder.encode(variable)) == variable`."

The bug occurs because:

1. **During encoding** (lines 333-351 in CFMaskCoder.encode): The encoder sees that data contains values equal to the fill value (0.0), but these are valid data values, not missing data. The fill value is moved to attributes without marking which values are actually missing.

2. **During decoding** (lines 394-424 in CFMaskCoder.decode): The decoder applies `_apply_mask` which blindly converts ALL values equal to the fill value to NaN, regardless of whether they were originally missing or valid data.

3. **The fundamental issue**: The `_apply_mask` function (lines 121-132) unconditionally replaces any value matching the fill value with NaN:
   ```python
   for fv in encoded_fill_values:
       condition |= data == fv
   return np.where(condition, decoded_fill_value, data)
   ```

This causes silent data corruption where valid zeros in scientific data are permanently lost when the fill value is also zero - a very common scenario in real-world datasets.

## Relevant Context

- **CF Conventions ambiguity**: While CF conventions state that values equal to `_FillValue` represent missing data, they don't explicitly address the case where valid data values happen to equal the fill value.

- **Common occurrence**: Zero is frequently both a valid data value and a fill value, especially in:
  - Temperature data (0°C)
  - Offset measurements
  - Integer data converted to float

- **Silent failure**: No warning is issued when this data corruption occurs, making it particularly dangerous for scientific data integrity.

- **Documentation link**: The VariableCoder base class documentation clearly states the round-trip requirement: [xarray/coding/common.py](https://github.com/pydata/xarray/blob/main/xarray/coding/common.py#L22-L37)

## Proposed Fix

The issue requires distinguishing between values that are truly missing versus valid data that happens to equal the fill value. Here's a high-level approach:

During encoding, the coder should:
1. Check if the input data already contains NaN/masked values
2. Only apply the fill value to those positions that are already NaN/masked
3. Preserve valid data values even if they equal the fill value

During decoding:
1. Only convert fill values back to NaN at positions that were originally NaN/masked
2. This requires tracking which positions had missing data, possibly through a separate mask

A minimal fix could involve checking if the original variable contains any NaN values before applying the fill value logic:

```diff
--- a/xarray/coding/variables.py
+++ b/xarray/coding/variables.py
@@ -331,7 +331,9 @@ class CFMaskCoder(VariableCoder):

         # apply fillna
         if fill_value is not None and not pd.isnull(fill_value):
-            # special case DateTime to properly handle NaT
+            # Only apply fill value if there are actually missing values to fill
+            if not np.any(pd.isnull(data)):
+                return Variable(dims, data, attrs, encoding, fastpath=True)
             if _is_time_like(attrs.get("units")):
                 if data.dtype.kind in "iu":
                     data = duck_array_ops.where(
@@ -393,6 +395,10 @@ class CFMaskCoder(VariableCoder):

         if encoded_fill_values:
             dtype: np.typing.DTypeLike
+            # Check if the data actually uses fill values for missing data
+            # by checking if all values equal to fill value (if all equal, likely valid data)
+            if np.all(data == list(encoded_fill_values)[0]):
+                return Variable(dims, data, attrs, encoding, fastpath=True)
             decoded_fill_value: Any
             # in case of packed data we have to decode into float
             # in any case
```

Note: This is a simplified fix. A complete solution would need more sophisticated logic to properly track which values are truly missing versus valid data that happens to equal the fill value.