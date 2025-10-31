# Bug Report: pandas.DataFrame.T Loses Dtype Information on Mixed-Type DataFrames

**Target**: `pandas.core.internals` (specifically DataFrame transpose operation)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When transposing a DataFrame with mixed dtypes (e.g., int64 and float64), the transpose operation loses dtype information. Specifically, `df.T.T` does not preserve the original dtypes, violating the mathematical property that transpose is self-inverse.

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

    assert df.shape == df_tt.shape

    for col in df.columns:
        assert df[col].dtype == df_tt[col].dtype  # FAILS: int64 becomes float64
```

**Failing input**: `n_rows=1, n_cols=2` (with alternating float64 and int64 columns)

## Reproducing the Bug

```python
import pandas as pd

df = pd.DataFrame({'a': [1.5], 'b': [2]})

print("Original dtypes:")
print(df.dtypes)

print("\nAfter T.T dtypes:")
print(df.T.T.dtypes)

assert df['b'].dtype == df.T.T['b'].dtype
```

**Output:**
```
Original dtypes:
a    float64
b      int64
dtype: object

After T.T dtypes:
a    float64
b    float64
dtype: object

AssertionError
```

## Why This Is A Bug

1. **Mathematical property violated**: Transpose is mathematically a self-inverse operation. For any matrix M, (M^T)^T = M. In pandas terms, `df.T.T` should equal `df` exactly, including dtypes.

2. **Silent data corruption**: Users performing `df.T.T` expect to get back their original DataFrame unchanged. The silent conversion from int64 to float64 can cause issues downstream, especially in contexts where integer dtypes are semantically important (e.g., IDs, counts, categorical codes).

3. **Inconsistent with pandas philosophy**: pandas generally preserves dtypes through operations unless there's an explicit reason to change them. Here, the dtype change happens silently without user action or warning.

## Root Cause

When pandas transposes a DataFrame with mixed dtypes, it must consolidate all columns into a single dtype for the intermediate transposed representation. The current implementation chooses float64 as the common dtype. When transposing back, pandas does not attempt to restore the original dtypes.

This happens in `pandas.core.internals` during the transpose operation, which uses the BlockManager to reorganize data. The BlockManager consolidates mixed-dtype blocks into a single float64 block during the first transpose, and this type information is not recovered on the second transpose.

## Fix

The fix would require pandas to either:

1. **Track original dtypes**: Store dtype metadata during the first transpose and use it to restore dtypes on the second transpose.

2. **Avoid dtype consolidation**: Implement a transpose mechanism that preserves block structure and doesn't force dtype consolidation.

3. **Explicit casting**: On the return transpose, attempt to downcast float64 columns back to their original dtypes when the conversion is lossless (e.g., 85.0 → 85 for int64).

The cleanest solution is (1), which would require changes to how pandas.core.internals handles transpose operations on mixed-dtype DataFrames, potentially storing dtype information in the transposed DataFrame's metadata.