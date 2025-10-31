# Bug Report: pandas.api.types.infer_dtype skipna parameter ignored for floats with None

**Target**: `pandas.api.types.infer_dtype`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `infer_dtype` is called with `skipna=True` on a list of floats containing `None`, it incorrectly returns `'mixed-integer-float'` instead of `'floating'`. The `skipna=True` parameter should cause `None` values to be ignored during type inference, making the result identical to the same list without `None` values.

## Property-Based Test

```python
import pandas as pd
from pandas.api.types import infer_dtype
from hypothesis import given, strategies as st, assume, settings


@settings(max_examples=500)
@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
                min_size=1, max_size=20))
def test_infer_dtype_skipna_consistency_floats(float_values):
    assume(len(float_values) > 0)

    result_without_none = infer_dtype(float_values, skipna=True)

    float_values_with_none = float_values + [None]
    result_with_none = infer_dtype(float_values_with_none, skipna=True)

    assert result_without_none == result_with_none, \
        f"infer_dtype with skipna=True should ignore None: " \
        f"{float_values} -> {result_without_none}, " \
        f"{float_values_with_none} -> {result_with_none}"
```

**Failing input**: `float_values=[0.0]`

## Reproducing the Bug

```python
from pandas.api.types import infer_dtype

pure_floats = [1.5, 2.5, 3.5]
floats_with_none = [1.5, None, 3.5]

result_pure = infer_dtype(pure_floats, skipna=True)
result_with_none = infer_dtype(floats_with_none, skipna=True)

print(f"infer_dtype([1.5, 2.5, 3.5], skipna=True) = '{result_pure}'")
print(f"infer_dtype([1.5, None, 3.5], skipna=True) = '{result_with_none}'")

assert result_pure == result_with_none, \
    f"Expected both to return '{result_pure}', but got '{result_with_none}'"
```

Output:
```
infer_dtype([1.5, 2.5, 3.5], skipna=True) = 'floating'
infer_dtype([1.5, None, 3.5], skipna=True) = 'mixed-integer-float'
AssertionError: Expected both to return 'floating', but got 'mixed-integer-float'
```

## Why This Is A Bug

The `skipna` parameter is documented as "Ignore NaN values when inferring the type" (with default `True`). When set to `True`, `None` values should be ignored during type inference. However, the function returns different results for `[1.5, 2.5, 3.5]` (returns `'floating'`) versus `[1.5, None, 3.5]` (returns `'mixed-integer-float'`) even though both are called with `skipna=True`.

This violates the documented contract and creates inconsistent behavior that would confuse users and lead to incorrect downstream logic based on the inferred type.

## Fix

The bug is likely in the C extension implementation of `infer_dtype`. The function needs to properly filter out `None` values before performing type inference when `skipna=True`. Without access to the C source, a high-level fix would be:

1. When `skipna=True`, filter out all `None`/`NaN` values from the input before type inference
2. Perform type inference on the filtered values
3. Return the inferred type

The current implementation appears to be partially considering `None` values even when `skipna=True`, causing it to return `'mixed-integer-float'` instead of the correct `'floating'` type.