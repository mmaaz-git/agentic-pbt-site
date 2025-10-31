# Bug Report: pandas.DataFrame.T Loses Integer Dtype During Double Transpose on Mixed Numeric DataFrames

**Target**: `pandas.core.frame.DataFrame.transpose`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When transposing a DataFrame containing both int64 and float64 columns, the double transpose operation (df.T.T) silently converts all int64 columns to float64, violating the mathematical property that transpose should be self-inverse.

## Property-Based Test

```python
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings

@given(
    n_rows=st.integers(min_value=1, max_value=20),
    n_cols=st.integers(min_value=1, max_value=5)
)
@settings(max_examples=200)
def test_transpose_preserves_dtypes(n_rows, n_cols):
    # Create dictionary of dtypes alternating between float64 and int64
    dtypes_dict = {f'col_{i}': np.float64 if i % 2 == 0 else np.int64 for i in range(n_cols)}

    data = {}
    for col, dtype in dtypes_dict.items():
        if dtype == np.float64:
            data[col] = np.random.randn(n_rows)
        else:
            data[col] = np.random.randint(0, 100, n_rows)

    df = pd.DataFrame(data)
    df_t = df.T
    df_tt = df_t.T

    # Check that shape is preserved
    assert df.shape == df_tt.shape, f"Shape mismatch: {df.shape} != {df_tt.shape}"

    # Check that dtypes are preserved
    for col in df.columns:
        original_dtype = df[col].dtype
        transposed_dtype = df_tt[col].dtype
        assert original_dtype == transposed_dtype, \
            f"Dtype mismatch for column '{col}': original={original_dtype}, after T.T={transposed_dtype}"

if __name__ == "__main__":
    # Run the test
    test_transpose_preserves_dtypes()
```

<details>

<summary>
**Failing input**: `n_rows=1, n_cols=2`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/4/hypo.py", line 37, in <module>
    test_transpose_preserves_dtypes()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/4/hypo.py", line 6, in test_transpose_preserves_dtypes
    n_rows=st.integers(min_value=1, max_value=20),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/4/hypo.py", line 32, in test_transpose_preserves_dtypes
    assert original_dtype == transposed_dtype, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Dtype mismatch for column 'col_1': original=int64, after T.T=float64
Falsifying example: test_transpose_preserves_dtypes(
    n_rows=1,
    n_cols=2,
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import numpy as np

# Create a DataFrame with mixed dtypes (float64 and int64)
df = pd.DataFrame({'a': [1.5], 'b': [2]})

print("Original DataFrame:")
print(df)
print("\nOriginal dtypes:")
print(df.dtypes)
print("\nOriginal 'b' column values and dtype:")
print(f"Values: {df['b'].values}")
print(f"Dtype: {df['b'].dtype}")

# Transpose the DataFrame
df_t = df.T
print("\n\nAfter first transpose (df.T):")
print(df_t)
print("\nDtypes after first transpose:")
print(df_t.dtypes)

# Transpose again to get back to original
df_tt = df_t.T
print("\n\nAfter double transpose (df.T.T):")
print(df_tt)
print("\nDtypes after double transpose:")
print(df_tt.dtypes)
print("\n'b' column after double transpose:")
print(f"Values: {df_tt['b'].values}")
print(f"Dtype: {df_tt['b'].dtype}")

# Check if dtypes are preserved
print("\n\nComparison:")
print(f"Original 'b' dtype: {df['b'].dtype}")
print(f"After T.T 'b' dtype: {df_tt['b'].dtype}")
print(f"Are they equal? {df['b'].dtype == df_tt['b'].dtype}")

# This assertion will fail
try:
    assert df['b'].dtype == df_tt['b'].dtype, f"Expected {df['b'].dtype} but got {df_tt['b'].dtype}"
    print("\nAssertion passed: dtypes are preserved")
except AssertionError as e:
    print(f"\nAssertion failed: {e}")
```

<details>

<summary>
AssertionError: Integer column dtype changes from int64 to float64 after double transpose
</summary>
```
Original DataFrame:
     a  b
0  1.5  2

Original dtypes:
a    float64
b      int64
dtype: object

Original 'b' column values and dtype:
Values: [2]
Dtype: int64


After first transpose (df.T):
     0
a  1.5
b  2.0

Dtypes after first transpose:
0    float64
dtype: object


After double transpose (df.T.T):
     a    b
0  1.5  2.0

Dtypes after double transpose:
a    float64
b    float64
dtype: object

'b' column after double transpose:
Values: [2.]
Dtype: float64


Comparison:
Original 'b' dtype: int64
After T.T 'b' dtype: float64
Are they equal? False

Assertion failed: Expected int64 but got float64
```
</details>

## Why This Is A Bug

This behavior violates several important principles and expectations:

1. **Mathematical Invariance Violated**: In linear algebra, the transpose operation is self-inverse, meaning (A^T)^T = A. Users reasonably expect `df.T.T` to return an identical DataFrame to the original, including data types. The current behavior breaks this fundamental mathematical property.

2. **Silent Type Coercion**: The conversion from int64 to float64 happens silently without any warning to the user. This is particularly problematic because:
   - Integer types often carry semantic meaning (IDs, counts, categorical codes)
   - Large integers (>2^53) lose precision when converted to float64
   - Operations that expect integer types may fail or produce unexpected results

3. **Inconsistent with Documentation**: The pandas documentation for `DataFrame.transpose()` states that "A copy is always required for mixed dtype DataFrames" and implies that mixed dtypes become object dtype, but this is incorrect for numeric mixed types. The actual behavior converts mixed numeric types to the most general numeric type (float64).

4. **Data Integrity Issues**: This bug can cause downstream problems in data pipelines where:
   - Integer columns represent discrete quantities that shouldn't have decimal values
   - Database schemas expect integer types for certain columns
   - Statistical operations assume integer distributions

5. **No Workaround in API**: There's no parameter in the transpose method to preserve dtypes, forcing users to manually track and restore dtypes after transposition.

## Relevant Context

- **Pandas Version**: 2.3.2
- **Python Version**: 3.13
- **Affected DataFrames**: Any DataFrame with mixed numeric dtypes (int8, int16, int32, int64 combined with float32 or float64)
- **Root Cause**: When pandas transposes a DataFrame with mixed dtypes, it consolidates blocks in the BlockManager. For mixed numeric types, it chooses float64 as the common dtype to avoid data loss. However, the original dtype information is not preserved for restoration during the second transpose.

The behavior can be traced to how pandas handles the internal BlockManager during transpose operations. The consolidation happens in `pandas.core.internals` where mixed-type blocks are merged into a single float64 block for efficiency, but the metadata about original dtypes is discarded.

## Proposed Fix

A high-level approach to fixing this issue would involve preserving dtype metadata during transpose operations:

1. **Store Original Dtypes**: When transposing a DataFrame with mixed numeric types, store the original column dtypes as metadata in the transposed DataFrame's attributes.

2. **Restore Dtypes on Re-transpose**: When transposing a DataFrame that has stored dtype metadata, attempt to restore the original dtypes where the conversion is lossless.

3. **Add Preserve Parameter**: Add an optional `preserve_dtypes` parameter to the transpose method that controls this behavior.

The implementation would require modifications to:
- `pandas/core/frame.py`: Modify the `transpose` method to handle dtype preservation
- `pandas/core/internals/managers.py`: Update BlockManager to track and restore original dtypes
- Add logic to detect when values can be safely converted back to their original integer types

This fix would ensure that `df.T.T` returns a DataFrame identical to `df` in both values and dtypes, maintaining the mathematical property of transpose being self-inverse while preserving data integrity.