# Bug Report: scipy.io.hb_write Round-Trip Failure

**Target**: `scipy.io.hb_write` / `scipy.io.hb_read`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `hb_write` function produces Harwell-Boeing format files that cannot be read back by `hb_read`, breaking the fundamental round-trip property. The bug occurs when a sparse matrix contains data values where a negative number appears after a positive number in the same row/column. The root cause is incorrect Fortran format to Python format conversion that produces values without adequate spacing.

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
```

**Failing input**:
```python
rows=1, cols=2, values=[0.0, -0.0]
```

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
except ValueError as e:
    print(f"Error: {e}")
finally:
    if os.path.exists(temp_path):
        os.unlink(temp_path)
```

Output:
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

## Why This Is A Bug

The file written by `hb_write` contains values concatenated without proper spacing:
```
 0.0000000000000000E+00-0.0000000000000000E+00
```

There is no space between the two exponential numbers. The `hb_read` function expects space-separated values and fails to parse this line.

The root cause is in `scipy/io/_harwell_boeing/_fortran_format_parser.py` line 166:

```python
@property
def python_format(self):
    return "%" + str(self.width-1) + "." + str(self.significand) + "E"
```

The Fortran format `E24.16` is converted to Python format `%23.16E` (using `width-1`). This produces 23-character fields instead of 24:
- Positive: ` 0.0000000000000000E+00` (23 chars, leading space)
- Negative: `-0.0000000000000000E+00` (23 chars, leading minus)

When concatenated: ` 0.0000000000000000E+00-0.0000000000000000E+00` (no space between values).

## Fix

The issue is the `width-1` adjustment in the `python_format` property. However, simply removing the -1 may break other functionality, as the adjustment might be intentional for some reason not apparent from the code.

The safest fix is to modify the `write_array` function in `scipy/io/_harwell_boeing/hb.py` to add explicit spacing between values:

```diff
--- a/scipy/io/_harwell_boeing/hb.py
+++ b/scipy/io/_harwell_boeing/hb.py
@@ -329,13 +329,13 @@ def _write_data(m, fid, header):
         # ar_nlines is the number of full lines, n is the number of items per
         # line, ffmt the fortran format
         pyfmt = fmt.python_format
-        pyfmt_full = pyfmt * fmt.repeat
+        pyfmt_full = ' '.join([pyfmt] * fmt.repeat)

         # for each array to write, we first write the full lines, and special
         # case for partial line
         full = ar[:(nlines - 1) * fmt.repeat]
         for row in full.reshape((nlines-1, fmt.repeat)):
             f.write(pyfmt_full % tuple(row) + "\n")
         nremain = ar.size - full.size
         if nremain > 0:
-            f.write((pyfmt * nremain) % tuple(ar[ar.size - nremain:]) + "\n")
+            f.write(' '.join([pyfmt] * nremain) % tuple(ar[ar.size - nremain:]) + "\n")
```

Alternatively, if the `width-1` adjustment is not essential, remove it:

```diff
--- a/scipy/io/_harwell_boeing/_fortran_format_parser.py
+++ b/scipy/io/_harwell_boeing/_fortran_format_parser.py
@@ -163,7 +163,7 @@ class ExpFormat:

     @property
     def python_format(self):
-        return "%" + str(self.width-1) + "." + str(self.significand) + "E"
+        return "%" + str(self.width) + "." + str(self.significand) + "E"
```