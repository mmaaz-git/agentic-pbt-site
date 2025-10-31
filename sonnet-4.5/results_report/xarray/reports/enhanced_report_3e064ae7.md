# Bug Report: xarray.core.indexes.normalize_label dtype type handling crash

**Target**: `xarray.core.indexes.normalize_label`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The internal `normalize_label` function crashes with an `AttributeError` when passed a numpy dtype type (e.g., `np.float32`) instead of a dtype instance (e.g., `np.dtype('float32')`), violating standard NumPy conventions.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import numpy as np
from xarray.core.indexes import normalize_label

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100),
                min_size=1, max_size=20, unique=True))
def test_normalize_label_with_float32_dtype(float_values):
    assume(len(float_values) > 0)
    float32_arr = np.array(float_values, dtype=np.float32)
    result = normalize_label(float32_arr, dtype=np.float32)
    assert result.dtype == np.float32

if __name__ == "__main__":
    test_normalize_label_with_float32_dtype()
```

<details>

<summary>
**Failing input**: `float_values=[0.0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 14, in <module>
    test_normalize_label_with_float32_dtype()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 6, in test_normalize_label_with_float32_dtype
    min_size=1, max_size=20, unique=True))
    ^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 10, in test_normalize_label_with_float32_dtype
    result = normalize_label(float32_arr, dtype=np.float32)
  File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/core/indexes.py", line 610, in normalize_label
    if dtype is not None and dtype.kind == "f" and value.dtype.kind != "b":
                             ^^^^^^^^^^
AttributeError: type object 'numpy.float32' has no attribute 'kind'
Falsifying example: test_normalize_label_with_float32_dtype(
    float_values=[0.0],  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from xarray.core.indexes import normalize_label

# Minimal reproduction of the bug
values = np.array([1.0, 2.0, 3.0], dtype=np.float32)

# This should work like NumPy's convention but crashes
result = normalize_label(values, dtype=np.float32)
print("Result:", result)
```

<details>

<summary>
AttributeError: type object 'numpy.float32' has no attribute 'kind'
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/33/repo.py", line 8, in <module>
    result = normalize_label(values, dtype=np.float32)
  File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/core/indexes.py", line 610, in normalize_label
    if dtype is not None and dtype.kind == "f" and value.dtype.kind != "b":
                             ^^^^^^^^^^
AttributeError: type object 'numpy.float32' has no attribute 'kind'
```
</details>

## Why This Is A Bug

This violates NumPy's universal convention of accepting both dtype types and dtype instances. NumPy functions like `np.array()` accept both forms interchangeably - for example, `np.array([1,2,3], dtype=np.float32)` works correctly. The function `normalize_label` internally calls `np.asarray(value, dtype=dtype)` at line 615, which itself accepts both dtype forms, creating an inconsistency where the wrapper function is more restrictive than the NumPy function it wraps.

The crash occurs because the code directly accesses `dtype.kind` at line 610 without first ensuring `dtype` is a proper dtype instance. NumPy dtype types like `np.float32` are type objects that don't have a `.kind` attribute, while dtype instances like `np.dtype(np.float32)` do have this attribute.

## Relevant Context

The `normalize_label` function is an internal utility function in xarray, not part of the public API. All internal xarray code that calls this function passes proper dtype instances obtained from `get_valid_numpy_dtype()` or stored dtype attributes like `self.coord_dtype`, so this bug doesn't affect normal xarray usage.

The function is called in three places within `xarray/core/indexes.py`:
- Line 842: `normalize_label(label, dtype=self.coord_dtype)`
- Line 1290: `normalize_label(v, dtype=self.level_coords_dtype[k])`
- Line 1350: `normalize_label(label)` (no dtype parameter)

All these internal calls use dtype instances, not dtype types, which is why this bug hasn't surfaced in regular xarray operations. However, the function could be called during debugging, testing, or by users exploring xarray internals.

## Proposed Fix

```diff
 def normalize_label(value, dtype=None) -> np.ndarray:
     if getattr(value, "ndim", 1) <= 1:
         value = _asarray_tuplesafe(value)
+    if dtype is not None:
+        dtype = np.dtype(dtype)
     if dtype is not None and dtype.kind == "f" and value.dtype.kind != "b":
         # pd.Index built from coordinate with float precision != 64
         # see https://github.com/pydata/xarray/pull/3153 for details
         # bypass coercing dtype for boolean indexers (ignore index)
         # see https://github.com/pydata/xarray/issues/5727
         value = np.asarray(value, dtype=dtype)
     return value
```