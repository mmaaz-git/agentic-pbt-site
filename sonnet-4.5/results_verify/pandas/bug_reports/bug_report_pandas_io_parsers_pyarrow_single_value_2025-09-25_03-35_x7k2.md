# Bug Report: pandas.io.parsers PyArrow Engine Single-Value CSV Parsing

**Target**: `pandas.io.parsers.read_csv` with `engine='pyarrow'`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The PyArrow engine fails to parse valid single-value CSV files (single row, single column) that are successfully parsed by both the C and Python engines, violating the expected engine equivalence property.

## Property-Based Test

```python
from io import StringIO
from hypothesis import given, strategies as st, assume, settings
import pandas.io.parsers as parsers
import pandas as pd

@settings(max_examples=100)
@given(
    data=st.lists(
        st.lists(st.integers(0, 100), min_size=1, max_size=5),
        min_size=1,
        max_size=20
    )
)
def test_engine_equivalence_with_pyarrow(data):
    num_cols = len(data[0])
    assume(all(len(row) == num_cols for row in data))

    csv_content = "\n".join(",".join(str(val) for val in row) for row in data)

    df_c = parsers.read_csv(StringIO(csv_content), header=None, engine='c')
    df_pyarrow = parsers.read_csv(StringIO(csv_content), header=None, engine='pyarrow')

    pd.testing.assert_frame_equal(df_c, df_pyarrow, check_dtype=False)
```

**Failing input**: `data=[[0]]` (CSV content: `"0"`)

## Reproducing the Bug

```python
from io import StringIO
import pandas as pd

csv_content = "0"

df_c = pd.read_csv(StringIO(csv_content), header=None, engine='c')
print("C engine result:")
print(df_c)
print(f"Shape: {df_c.shape}")

df_python = pd.read_csv(StringIO(csv_content), header=None, engine='python')
print("\nPython engine result:")
print(df_python)
print(f"Shape: {df_python.shape}")

df_pyarrow = pd.read_csv(StringIO(csv_content), header=None, engine='pyarrow')
```

**Expected output for all engines:**
```
   0
0  0
Shape: (1, 1)
```

**Actual output:**
- C engine: ✓ Success (1, 1)
- Python engine: ✓ Success (1, 1)
- PyArrow engine: ✗ `pandas.errors.ParserError: CSV parse error: Empty CSV file or block: cannot infer number of columns`

## Why This Is A Bug

1. **Valid input**: The string `"0"` is a valid CSV file containing a single value. It follows standard CSV format.

2. **Engine inconsistency**: Both the C and Python engines successfully parse this input, producing a DataFrame with shape (1, 1) and value 0. The PyArrow engine fails, violating the documented engine equivalence.

3. **Incorrect error message**: The error claims the CSV is "empty," which is false—it contains the value `"0"`.

4. **Property violation**: This violates the **multi-implementation equivalence property**. The pandas documentation states that different engines should produce the same results for supported features, though PyArrow is "experimental."

5. **Real-world impact**: Users attempting to parse simple single-value CSV files with the PyArrow engine will encounter unexpected crashes. This could affect:
   - Automated data pipelines that process CSV files of varying sizes
   - Users testing PyArrow's performance on simple inputs
   - Migration from C/Python engines to PyArrow

6. **Scope of the bug**: Testing reveals this affects all single-value CSV files, regardless of the actual value:
   - `"0"` → fails
   - `"42"` → fails
   - `"hello"` → fails
   - `"3.14"` → fails

## Fix

This appears to be a bug in the PyArrow CSV parsing library's handling of single-line CSV files without a trailing newline.

**Potential fixes:**

1. **Upstream fix (preferred)**: Report and fix this in the PyArrow library itself, where the CSV parser should handle single-value files consistently with other parsers.

2. **Pandas workaround**: Add special handling in pandas for single-line CSV when `engine='pyarrow'`:
   - Detect single-line CSV input without a trailing newline
   - Add a newline character before passing to PyArrow
   - Or fall back to C/Python engine for this edge case

3. **Better error handling**: At minimum, provide a clearer error message explaining the limitation and suggesting a workaround (e.g., "PyArrow engine does not support single-value CSV files. Use engine='c' or engine='python' instead").

**Example workaround for users:**
```python
df = pd.read_csv(StringIO(csv_content + "\n"), header=None, engine='pyarrow')
```

Or simply use a different engine for small CSV files:
```python
df = pd.read_csv(StringIO(csv_content), header=None, engine='c')
```