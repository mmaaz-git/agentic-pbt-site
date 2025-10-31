# Bug Report: pandas.core.algorithms.factorize Big-Endian Array Support

**Target**: `pandas.core.algorithms.factorize`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`pandas.core.algorithms.factorize()` crashes with a ValueError when given numpy arrays with big-endian byte order, even though pandas generally supports big-endian arrays (Series and DataFrame constructors accept them without issue).

## Property-Based Test

```python
import numpy as np
import pandas.core.algorithms as algorithms
from hypothesis import given
from hypothesis.extra import numpy as npst

@given(npst.arrays(dtype=npst.integer_dtypes() | npst.floating_dtypes(),
                   shape=npst.array_shapes(max_dims=1)))
def test_factorize_round_trip(arr):
    codes, uniques = algorithms.factorize(arr)
    assert len(codes) == len(arr)
```

**Failing input**: `array([0], dtype='>i8')` (and similar for `>i4`, `>i2`, `>f8`, `>f4`)

## Reproducing the Bug

```python
import numpy as np
import pandas.core.algorithms as algorithms

arr_big_endian = np.array([1, 2, 3, 2, 1], dtype='>i8')
codes, uniques = algorithms.factorize(arr_big_endian)
```

Output:
```
ValueError: Big-endian buffer not supported on little-endian compiler
```

## Why This Is A Bug

1. **Pandas generally supports big-endian arrays**: Series and DataFrame constructors accept big-endian arrays without error
   ```python
   >>> arr = np.array([1, 2, 3], dtype='>i8')
   >>> pd.Series(arr)  # Works fine
   0    1
   1    2
   2    3
   dtype: >i8
   ```

2. **Real-world use case**: Big-endian arrays can arise from:
   - Reading binary data from network protocols
   - Reading files created on big-endian systems
   - Cross-platform data processing

3. **NumPy supports both byte orders**: Since NumPy allows creation and manipulation of big-endian arrays, pandas should handle them consistently

The error originates in the Cython hashtable implementation (`pandas/_libs/hashtable_class_helper.pxi`), which doesn't handle non-native byte order.

## Fix

The fix should convert arrays to native byte order before passing them to the hashtable. In `pandas/core/algorithms.py`, in the `factorize` or `factorize_array` function:

```diff
def factorize_array(
    values: np.ndarray,
    use_na_sentinel: bool = True,
    size_hint: int | None = None,
    na_value: object = None,
    mask: npt.NDArray[np.bool_] | None = None,
) -> tuple[npt.NDArray[np.intp], np.ndarray]:
+    # Convert to native byte order if needed
+    if values.dtype.byteorder in ('>', '<'):
+        values = values.astype(values.dtype.newbyteorder('='))
+
    original = values
    ...
```

This mirrors how pandas handles byte order in other parts of the codebase and ensures compatibility with the Cython hashtable implementation.