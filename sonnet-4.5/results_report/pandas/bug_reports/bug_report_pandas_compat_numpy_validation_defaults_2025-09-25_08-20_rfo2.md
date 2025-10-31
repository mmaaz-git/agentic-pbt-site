# Bug Report: pandas.compat.numpy Validation Defaults Include Non-Existent Parameters

**Target**: `pandas.compat.numpy.function` (MEDIAN_DEFAULTS, MEAN_DEFAULTS, MINMAX_DEFAULTS)
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

Multiple validation default dictionaries incorrectly include parameters that don't exist in the corresponding numpy function signatures, causing pandas to silently accept invalid parameters instead of raising TypeError.

## Property-Based Test

```python
import pytest
import numpy as np
import inspect
from hypothesis import given, strategies as st
from pandas.compat.numpy.function import MEDIAN_DEFAULTS, MEAN_DEFAULTS, MINMAX_DEFAULTS


@given(param_name=st.sampled_from(list(MEDIAN_DEFAULTS.keys())))
def test_median_defaults_match_numpy(param_name):
    numpy_params = set(inspect.signature(np.median).parameters.keys()) - {'a'}
    assert param_name in numpy_params


@given(param_name=st.sampled_from(list(MEAN_DEFAULTS.keys())))
def test_mean_defaults_match_numpy(param_name):
    numpy_params = set(inspect.signature(np.mean).parameters.keys()) - {'a'}
    assert param_name in numpy_params


@given(param_name=st.sampled_from(list(MINMAX_DEFAULTS.keys())))
def test_minmax_defaults_match_numpy(param_name):
    numpy_params = set(inspect.signature(np.min).parameters.keys()) - {'a'}
    assert param_name in numpy_params
```

**Failing inputs**:
- `test_median_defaults_match_numpy`: param_name='dtype'
- `test_mean_defaults_match_numpy`: param_name='initial'
- `test_minmax_defaults_match_numpy`: param_name='dtype'

## Reproducing the Bug

```python
import pandas as pd
import numpy as np
import inspect

series = pd.Series([1, 2, 3])

print(f"numpy.median signature: {inspect.signature(np.median)}")
result = series.median(dtype=None)
print(f"series.median(dtype=None) = {result}")

print(f"
numpy.mean signature: {inspect.signature(np.mean)}")
result = series.mean(initial=None)
print(f"series.mean(initial=None) = {result}")

print(f"
numpy.min signature: {inspect.signature(np.min)}")
result = series.min(dtype=None)
print(f"series.min(dtype=None) = {result}")

print(f"
numpy.max signature: {inspect.signature(np.max)}")
result = series.max(dtype=None)
print(f"series.max(dtype=None) = {result}")
```

Output:
```
numpy.median signature: (a, axis=None, out=None, overwrite_input=False, keepdims=False)
series.median(dtype=None) = 2.0

numpy.mean signature: (a, axis=None, dtype=None, out=None, keepdims=<no value>, *, where=<no value>)
series.mean(initial=None) = 2.0

numpy.min signature: (a, axis=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)
series.min(dtype=None) = 1

numpy.max signature: (a, axis=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)
series.max(dtype=None) = 3
```

All calls are silently accepted when they should raise TypeError for unexpected keyword arguments.

## Why This Is A Bug

The validation defaults are meant to ensure pandas methods reject numpy parameters that:
1. Don't exist in numpy's signature, OR
2. Are passed with non-default values

The following incorrect entries were found:

1. **MEDIAN_DEFAULTS includes 'dtype'**: numpy.median has no dtype parameter
2. **MEAN_DEFAULTS includes 'initial'**: numpy.mean has no initial parameter
3. **MINMAX_DEFAULTS includes 'dtype'**: numpy.min/max have no dtype parameter

This causes pandas to:
- Silently accept these parameters when set to None (incorrect permissive behavior)
- Raise ValueError with misleading message for non-None values (should be TypeError about unexpected keyword)

## Fix

```diff
--- a/pandas/compat/numpy/function.py
+++ b/pandas/compat/numpy/function.py
@@ -256,13 +256,11 @@ validate_logical_func = CompatValidator(LOGICAL_FUNC_DEFAULTS, method="kwargs")

-MINMAX_DEFAULTS = {"axis": None, "dtype": None, "out": None, "keepdims": False}
+MINMAX_DEFAULTS = {"axis": None, "out": None, "keepdims": False}
 validate_min = CompatValidator(
     MINMAX_DEFAULTS, fname="min", method="both", max_fname_arg_count=1
 )

 SUM_DEFAULTS = STAT_FUNC_DEFAULTS.copy()
 SUM_DEFAULTS["axis"] = None
 SUM_DEFAULTS["keepdims"] = False
-SUM_DEFAULTS["initial"] = None

 PROD_DEFAULTS = SUM_DEFAULTS.copy()

 MEAN_DEFAULTS = SUM_DEFAULTS.copy()
+MEAN_DEFAULTS.pop("initial", None)

-MEDIAN_DEFAULTS = STAT_FUNC_DEFAULTS.copy()
+MEDIAN_DEFAULTS = {"out": None, "overwrite_input": False, "keepdims": False}
-MEDIAN_DEFAULTS["overwrite_input"] = False
-MEDIAN_DEFAULTS["keepdims"] = False
```

Note: Missing 'where' parameter for sum/prod/mean/min/max may be intentional if pandas doesn't support it, so not fixed here.
