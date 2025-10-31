# Bug Report: scipy.io.hb_write Round-Trip Failure Due to Incorrect Fortran Format Conversion

**Target**: `scipy.io.hb_write` / `scipy.io.hb_read`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `hb_write` function produces invalid Harwell-Boeing format files that cannot be read back by `hb_read`, breaking the fundamental round-trip property. The bug occurs when writing sparse matrices where positive values are followed by negative values in the data array.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import tempfile
import os
import numpy as np
import scipy.io
from scipy.sparse import csc_array

@given(st.integers(min_value=1, max_value=100),
       st.integers(min_value=1, max_value=100),
       st.lists(st.floats(allow_nan=False, allow_infinity=False,
                          min_value=-1e10, max_value=1e10),
               min_size=1, max_size=500))
@settings(max_examples=50, deadline=None)
def test_harwell_boeing_round_trip(rows, cols, values):
    num_values = min(len(values), rows * cols)

    row_indices = np.random.choice(rows, size=num_values, replace=True)
    col_indices = np.random.choice(cols, size=num_values, replace=True)
    data_values = np.array(values[:num_values])

    original = csc_array((data_values, (row_indices, col_indices)),
                          shape=(rows, cols))

    with tempfile.NamedTemporaryFile(mode='w', suffix='.hb', delete=False) as f:
        temp_path = f.name

    try:
        scipy.io.hb_write(temp_path, original)
        recovered = scipy.io.hb_read(temp_path)

        assert recovered.shape == original.shape
        np.testing.assert_allclose(recovered.toarray(), original.toarray(), rtol=1e-6)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

if __name__ == "__main__":
    test_harwell_boeing_round_trip()
```

<details>

<summary>
**Failing input**: `rows=1, cols=2, values=[0.0, -0.0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo_hb.py", line 38, in <module>
    test_harwell_boeing_round_trip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo_hb.py", line 9, in test_harwell_boeing_round_trip
    st.integers(min_value=1, max_value=100),
            ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo_hb.py", line 29, in test_harwell_boeing_round_trip
    recovered = scipy.io.hb_read(temp_path)
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/_harwell_boeing/hb.py", line 510, in hb_read
    data = _get_matrix(f)
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/_harwell_boeing/hb.py", line 504, in _get_matrix
    return hb.read_matrix()
           ~~~~~~~~~~~~~~^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/_harwell_boeing/hb.py", line 455, in read_matrix
    return _read_hb_data(self._fid, self._hb_info)
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/_harwell_boeing/hb.py", line 319, in _read_hb_data
    val = np.fromstring(val_string,
            dtype=header.values_dtype, sep=' ')
ValueError: string or file could not be read to its end due to unmatched data
Falsifying example: test_harwell_boeing_round_trip(
    rows=1,
    cols=2,
    values=[0.0, -0.0],
)
```
</details>

## Reproducing the Bug

```python
import tempfile
import os
import numpy as np
import scipy.io
from scipy.sparse import csc_array

row_indices = np.array([0, 0])
col_indices = np.array([0, 1])
data_values = np.array([0.0, -0.0])

original = csc_array((data_values, (row_indices, col_indices)), shape=(1, 2))

with tempfile.NamedTemporaryFile(mode='w', suffix='.hb', delete=False) as f:
    temp_path = f.name

try:
    scipy.io.hb_write(temp_path, original)

    with open(temp_path, 'r') as f:
        print("File contents:")
        print(f.read())

    recovered = scipy.io.hb_read(temp_path)
    print("Successfully read back the file!")
    print(f"Original shape: {original.shape}, Recovered shape: {recovered.shape}")
    np.testing.assert_allclose(recovered.toarray(), original.toarray(), rtol=1e-6)
    print("Data matches!")
except ValueError as e:
    print(f"Error: {e}")
finally:
    if os.path.exists(temp_path):
        os.unlink(temp_path)
```

<details>

<summary>
Error: string or file could not be read to its end due to unmatched data
</summary>
```
File contents:
Default title                                                           0
             3             1             1             1
RUA                        1             2             2             0
(40I2)          (40I2)          (3E24.16)
 1 2 3
 1 1
 0.0000000000000000E+00-0.0000000000000000E+00

Error: string or file could not be read to its end due to unmatched data
```
</details>

## Why This Is A Bug

This violates expected behavior because:

1. **Round-trip guarantee broken**: The scipy documentation explicitly demonstrates that `hb_write` followed by `hb_read` should successfully recover the original data. Example code shows `hb_write("data.hb", data)` then `hb_read("data.hb")` should work.

2. **Format specification violation**: The Harwell-Boeing format requires Fortran format `E24.16` to produce fields exactly 24 characters wide. The bug causes scipy to generate 23-character fields, violating the standard. The written file contains `0.0000000000000000E+00-0.0000000000000000E+00` with no space between values.

3. **Affects common use cases**: The bug occurs whenever positive values are followed by negative values in the CSC data array, particularly with:
   - Zero and negative zero (0.0, -0.0)
   - Values with 3-digit exponents (e.g., 1e100, -1e100, 1e-100, -1e-100)
   - Any positive value immediately followed by a negative value when formatted with %23.16E

4. **Data integrity failure**: Users cannot reliably save and restore their sparse matrices using the Harwell-Boeing format, which is a standard format for sparse matrix exchange.

## Relevant Context

The root cause is in `scipy/io/_harwell_boeing/_fortran_format_parser.py` at line 166:

```python
@property
def python_format(self):
    return "%" + str(self.width-1) + "." + str(self.significand) + "E"
```

This incorrectly converts Fortran format `E24.16` to Python format `%23.16E` (using `width-1`). When Python formats values with `%23.16E`:
- Positive: ` 0.0000000000000000E+00` (23 chars with leading space)
- Negative: `-0.0000000000000000E+00` (23 chars with leading minus)

In `scipy/io/_harwell_boeing/hb.py` line 332, values are concatenated without explicit spacing:
```python
pyfmt_full = pyfmt * fmt.repeat  # Creates '%23.16E%23.16E%23.16E'
```

This results in values being directly concatenated without proper field separation when a positive value is followed by a negative value.

Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.hb_write.html
Harwell-Boeing specification: https://math.nist.gov/MatrixMarket/formats.html#hb

## Proposed Fix

Remove the incorrect `width-1` adjustment in the `python_format` property:

```diff
--- a/scipy/io/_harwell_boeing/_fortran_format_parser.py
+++ b/scipy/io/_harwell_boeing/_fortran_format_parser.py
@@ -163,7 +163,7 @@ class ExpFormat:

     @property
     def python_format(self):
-        return "%" + str(self.width-1) + "." + str(self.significand) + "E"
+        return "%" + str(self.width) + "." + str(self.significand) + "E"
```