# Bug Report: pandas.compat.numpy Validation Defaults Include Non-Existent NumPy Parameters

**Target**: `pandas.compat.numpy.function` (MEDIAN_DEFAULTS, MEAN_DEFAULTS, MINMAX_DEFAULTS)
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The pandas.compat.numpy.function module incorrectly includes parameters in validation defaults that don't exist in the corresponding NumPy function signatures, causing pandas to silently accept invalid parameters instead of raising TypeError as it should.

## Property-Based Test

```python
import pytest
import numpy as np
import inspect
from hypothesis import given, strategies as st
import sys
sys.path.append('/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')
from pandas.compat.numpy.function import MEDIAN_DEFAULTS, MEAN_DEFAULTS, MINMAX_DEFAULTS


@given(param_name=st.sampled_from(list(MEDIAN_DEFAULTS.keys())))
def test_median_defaults_match_numpy(param_name):
    numpy_params = set(inspect.signature(np.median).parameters.keys()) - {'a'}
    assert param_name in numpy_params, f"Parameter '{param_name}' is in MEDIAN_DEFAULTS but not in numpy.median signature"


@given(param_name=st.sampled_from(list(MEAN_DEFAULTS.keys())))
def test_mean_defaults_match_numpy(param_name):
    numpy_params = set(inspect.signature(np.mean).parameters.keys()) - {'a'}
    assert param_name in numpy_params, f"Parameter '{param_name}' is in MEAN_DEFAULTS but not in numpy.mean signature"


@given(param_name=st.sampled_from(list(MINMAX_DEFAULTS.keys())))
def test_minmax_defaults_match_numpy(param_name):
    numpy_params = set(inspect.signature(np.min).parameters.keys()) - {'a'}
    assert param_name in numpy_params, f"Parameter '{param_name}' is in MINMAX_DEFAULTS but not in numpy.min signature"


if __name__ == "__main__":
    print("Testing MEDIAN_DEFAULTS against numpy.median signature...")
    print(f"MEDIAN_DEFAULTS keys: {list(MEDIAN_DEFAULTS.keys())}")
    print(f"numpy.median parameters: {list(inspect.signature(np.median).parameters.keys())}")
    try:
        test_median_defaults_match_numpy()
        print("✓ All MEDIAN_DEFAULTS parameters exist in numpy.median")
    except AssertionError as e:
        print(f"✗ Test failed: {e}")

    print("\nTesting MEAN_DEFAULTS against numpy.mean signature...")
    print(f"MEAN_DEFAULTS keys: {list(MEAN_DEFAULTS.keys())}")
    print(f"numpy.mean parameters: {list(inspect.signature(np.mean).parameters.keys())}")
    try:
        test_mean_defaults_match_numpy()
        print("✓ All MEAN_DEFAULTS parameters exist in numpy.mean")
    except AssertionError as e:
        print(f"✗ Test failed: {e}")

    print("\nTesting MINMAX_DEFAULTS against numpy.min signature...")
    print(f"MINMAX_DEFAULTS keys: {list(MINMAX_DEFAULTS.keys())}")
    print(f"numpy.min parameters: {list(inspect.signature(np.min).parameters.keys())}")
    try:
        test_minmax_defaults_match_numpy()
        print("✓ All MINMAX_DEFAULTS parameters exist in numpy.min")
    except AssertionError as e:
        print(f"✗ Test failed: {e}")
```

<details>

<summary>
**Failing input**: `dtype` for MEDIAN_DEFAULTS, `initial` for MEAN_DEFAULTS, `dtype` for MINMAX_DEFAULTS
</summary>
```
Testing MEDIAN_DEFAULTS against numpy.median signature...
MEDIAN_DEFAULTS keys: ['dtype', 'out', 'overwrite_input', 'keepdims']
numpy.median parameters: ['a', 'axis', 'out', 'overwrite_input', 'keepdims']
✗ Test failed: Parameter 'dtype' is in MEDIAN_DEFAULTS but not in numpy.median signature

Testing MEAN_DEFAULTS against numpy.mean signature...
MEAN_DEFAULTS keys: ['dtype', 'out', 'axis', 'keepdims', 'initial']
numpy.mean parameters: ['a', 'axis', 'dtype', 'out', 'keepdims', 'where']
✗ Test failed: Parameter 'initial' is in MEAN_DEFAULTS but not in numpy.mean signature

Testing MINMAX_DEFAULTS against numpy.min signature...
MINMAX_DEFAULTS keys: ['axis', 'dtype', 'out', 'keepdims']
numpy.min parameters: ['a', 'axis', 'out', 'keepdims', 'initial', 'where']
✗ Test failed: Parameter 'dtype' is in MINMAX_DEFAULTS but not in numpy.min signature
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import numpy as np
import inspect

# Create a simple pandas Series for testing
series = pd.Series([1, 2, 3])

# Test 1: numpy.median has NO dtype parameter, but pandas accepts dtype=None
print("Test 1: MEDIAN with dtype parameter")
print(f"numpy.median signature: {inspect.signature(np.median)}")
print("  - Note: numpy.median has NO dtype parameter")
try:
    result = series.median(dtype=None)
    print(f"  - series.median(dtype=None) = {result} (INCORRECTLY ACCEPTED)")
except Exception as e:
    print(f"  - series.median(dtype=None) raised: {type(e).__name__}: {e}")

try:
    result = series.median(dtype=float)
    print(f"  - series.median(dtype=float) = {result} (INCORRECTLY ACCEPTED)")
except Exception as e:
    print(f"  - series.median(dtype=float) raised: {type(e).__name__}: {e}")

# Test 2: numpy.mean has NO initial parameter, but pandas accepts initial=None
print("\nTest 2: MEAN with initial parameter")
print(f"numpy.mean signature: {inspect.signature(np.mean)}")
print("  - Note: numpy.mean has NO initial parameter")
try:
    result = series.mean(initial=None)
    print(f"  - series.mean(initial=None) = {result} (INCORRECTLY ACCEPTED)")
except Exception as e:
    print(f"  - series.mean(initial=None) raised: {type(e).__name__}: {e}")

try:
    result = series.mean(initial=1.0)
    print(f"  - series.mean(initial=1.0) = {result} (INCORRECTLY ACCEPTED)")
except Exception as e:
    print(f"  - series.mean(initial=1.0) raised: {type(e).__name__}: {e}")

# Test 3: numpy.min has NO dtype parameter, but pandas accepts dtype=None
print("\nTest 3: MIN with dtype parameter")
print(f"numpy.min signature: {inspect.signature(np.min)}")
print("  - Note: numpy.min has NO dtype parameter")
try:
    result = series.min(dtype=None)
    print(f"  - series.min(dtype=None) = {result} (INCORRECTLY ACCEPTED)")
except Exception as e:
    print(f"  - series.min(dtype=None) raised: {type(e).__name__}: {e}")

try:
    result = series.min(dtype=float)
    print(f"  - series.min(dtype=float) = {result} (INCORRECTLY ACCEPTED)")
except Exception as e:
    print(f"  - series.min(dtype=float) raised: {type(e).__name__}: {e}")

# Test 4: numpy.max has NO dtype parameter, but pandas accepts dtype=None
print("\nTest 4: MAX with dtype parameter")
print(f"numpy.max signature: {inspect.signature(np.max)}")
print("  - Note: numpy.max has NO dtype parameter")
try:
    result = series.max(dtype=None)
    print(f"  - series.max(dtype=None) = {result} (INCORRECTLY ACCEPTED)")
except Exception as e:
    print(f"  - series.max(dtype=None) raised: {type(e).__name__}: {e}")

try:
    result = series.max(dtype=float)
    print(f"  - series.max(dtype=float) = {result} (INCORRECTLY ACCEPTED)")
except Exception as e:
    print(f"  - series.max(dtype=float) raised: {type(e).__name__}: {e}")

# Demonstrate what SHOULD happen (TypeError for unexpected keyword)
print("\nExpected behavior demonstration:")
print("Calling numpy functions directly with invalid parameters:")
try:
    np.median([1, 2, 3], dtype=None)
except TypeError as e:
    print(f"  - np.median([1,2,3], dtype=None) raised: TypeError: {e}")

try:
    np.mean([1, 2, 3], initial=None)
except TypeError as e:
    print(f"  - np.mean([1,2,3], initial=None) raised: TypeError: {e}")

try:
    np.min([1, 2, 3], dtype=None)
except TypeError as e:
    print(f"  - np.min([1,2,3], dtype=None) raised: TypeError: {e}")
```

<details>

<summary>
Pandas incorrectly accepts invalid parameters with None values; raises ValueError instead of TypeError for non-None values
</summary>
```
Test 1: MEDIAN with dtype parameter
numpy.median signature: (a, axis=None, out=None, overwrite_input=False, keepdims=False)
  - Note: numpy.median has NO dtype parameter
  - series.median(dtype=None) = 2.0 (INCORRECTLY ACCEPTED)
  - series.median(dtype=float) raised: ValueError: the 'dtype' parameter is not supported in the pandas implementation of median()

Test 2: MEAN with initial parameter
numpy.mean signature: (a, axis=None, dtype=None, out=None, keepdims=<no value>, *, where=<no value>)
  - Note: numpy.mean has NO initial parameter
  - series.mean(initial=None) = 2.0 (INCORRECTLY ACCEPTED)
  - series.mean(initial=1.0) raised: ValueError: the 'initial' parameter is not supported in the pandas implementation of mean()

Test 3: MIN with dtype parameter
numpy.min signature: (a, axis=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)
  - Note: numpy.min has NO dtype parameter
  - series.min(dtype=None) = 1 (INCORRECTLY ACCEPTED)
  - series.min(dtype=float) raised: ValueError: the 'dtype' parameter is not supported in the pandas implementation of min()

Test 4: MAX with dtype parameter
numpy.max signature: (a, axis=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)
  - Note: numpy.max has NO dtype parameter
  - series.max(dtype=None) = 3 (INCORRECTLY ACCEPTED)
  - series.max(dtype=float) raised: ValueError: the 'dtype' parameter is not supported in the pandas implementation of max()

Expected behavior demonstration:
Calling numpy functions directly with invalid parameters:
  - np.median([1,2,3], dtype=None) raised: TypeError: median() got an unexpected keyword argument 'dtype'
  - np.mean([1,2,3], initial=None) raised: TypeError: mean() got an unexpected keyword argument 'initial'
  - np.min([1,2,3], dtype=None) raised: TypeError: min() got an unexpected keyword argument 'dtype'
```
</details>

## Why This Is A Bug

The pandas.compat.numpy.function module explicitly states in its docstring that validation exists "to make sure that any extra parameters passed correspond ONLY to those in the numpy signature." However, the validation defaults include parameters that don't exist in NumPy:

1. **MEDIAN_DEFAULTS includes 'dtype'**: NumPy's median function signature is `(a, axis=None, out=None, overwrite_input=False, keepdims=False)` - there is NO dtype parameter. Yet MEDIAN_DEFAULTS inherits from STAT_FUNC_DEFAULTS which includes `dtype=None`.

2. **MEAN_DEFAULTS includes 'initial'**: NumPy's mean function signature is `(a, axis=None, dtype=None, out=None, keepdims=<no value>, *, where=<no value>)` - there is NO initial parameter. Yet MEAN_DEFAULTS inherits from SUM_DEFAULTS which includes `initial=None`.

3. **MINMAX_DEFAULTS includes 'dtype'**: NumPy's min/max function signatures are `(a, axis=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)` - there is NO dtype parameter. Yet MINMAX_DEFAULTS explicitly includes `dtype=None`.

This violates the module's stated contract and causes incorrect behavior:
- Parameters that don't exist in NumPy are silently accepted when passed as None (should raise TypeError)
- Parameters that don't exist in NumPy raise ValueError with misleading message "not supported" when passed with non-None values (should raise TypeError for unexpected keyword)

## Relevant Context

The validation system in pandas.compat.numpy.function serves an important purpose: when pandas functions accept `**kwargs` to accommodate NumPy arguments, the validation ensures users don't abuse these parameters by passing invalid ones or non-default values.

The bug stems from inheritance patterns in the code:
- Line 283-286: `STAT_FUNC_DEFAULTS` includes `dtype=None`
- Line 287-291: `SUM_DEFAULTS` copies STAT_FUNC_DEFAULTS and adds `initial=None`
- Line 294: `MEAN_DEFAULTS` copies SUM_DEFAULTS (inheriting the incorrect 'initial')
- Line 296-298: `MEDIAN_DEFAULTS` copies STAT_FUNC_DEFAULTS (inheriting the incorrect 'dtype')
- Line 254: `MINMAX_DEFAULTS` explicitly includes `dtype=None`

Source code location: `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/compat/numpy/function.py`

## Proposed Fix

```diff
--- a/pandas/compat/numpy/function.py
+++ b/pandas/compat/numpy/function.py
@@ -251,7 +251,7 @@ validate_any = CompatValidator(
 LOGICAL_FUNC_DEFAULTS = {"out": None, "keepdims": False}
 validate_logical_func = CompatValidator(LOGICAL_FUNC_DEFAULTS, method="kwargs")

-MINMAX_DEFAULTS = {"axis": None, "dtype": None, "out": None, "keepdims": False}
+MINMAX_DEFAULTS = {"axis": None, "out": None, "keepdims": False}
 validate_min = CompatValidator(
     MINMAX_DEFAULTS, fname="min", method="both", max_fname_arg_count=1
 )
@@ -291,9 +291,12 @@ SUM_DEFAULTS["initial"] = None

 PROD_DEFAULTS = SUM_DEFAULTS.copy()

-MEAN_DEFAULTS = SUM_DEFAULTS.copy()
+MEAN_DEFAULTS = STAT_FUNC_DEFAULTS.copy()
+MEAN_DEFAULTS["axis"] = None
+MEAN_DEFAULTS["keepdims"] = False

-MEDIAN_DEFAULTS = STAT_FUNC_DEFAULTS.copy()
+MEDIAN_DEFAULTS = {}
+MEDIAN_DEFAULTS["out"] = None
 MEDIAN_DEFAULTS["overwrite_input"] = False
 MEDIAN_DEFAULTS["keepdims"] = False
```