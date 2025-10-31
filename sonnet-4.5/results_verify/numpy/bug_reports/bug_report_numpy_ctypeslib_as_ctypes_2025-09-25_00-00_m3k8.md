# Bug Report: numpy.ctypeslib.as_ctypes Structured Dtype Support

**Target**: `numpy.ctypeslib.as_ctypes`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`as_ctypes` fails to convert arrays with structured dtypes, even though `as_ctypes_type` supports them, because it uses the wrong field from `__array_interface__`.

## Property-Based Test

```python
import numpy as np
import numpy.ctypeslib as npc
from hypothesis import given, strategies as st, settings

@given(num_fields=st.integers(min_value=1, max_value=5))
@settings(max_examples=100)
def test_as_ctypes_structured_arrays(num_fields):
    field_types = ['i4', 'f8', 'u1', 'u2']
    fields = [(f'f{i}', field_types[i % len(field_types)]) for i in range(num_fields)]
    struct_dtype = np.dtype(fields)

    arr = np.zeros(10, dtype=struct_dtype)
    arr = np.ascontiguousarray(arr)
    arr.flags.writeable = True

    c_obj = npc.as_ctypes(arr)
    result = npc.as_array(c_obj)

    assert result.dtype == arr.dtype
    np.testing.assert_array_equal(result, arr)
```

**Failing input**: `num_fields=1`

## Reproducing the Bug

```python
import numpy as np
import numpy.ctypeslib as npc

struct_dtype = np.dtype([('x', 'i4'), ('y', 'f8')])
arr = np.zeros(5, dtype=struct_dtype)
arr = np.ascontiguousarray(arr)
arr.flags.writeable = True

ctype = npc.as_ctypes_type(struct_dtype)
print(f"as_ctypes_type works: {ctype}")

c_obj = npc.as_ctypes(arr)
```

Output:
```
as_ctypes_type works: <class 'struct'>
NotImplementedError: Converting dtype('V12') to a ctypes type
```

## Why This Is A Bug

`as_ctypes_type` correctly handles structured dtypes, but `as_ctypes` does not. The issue is that `as_ctypes` uses `__array_interface__["typestr"]` which is `|V12` for structured types (a void type), instead of using `__array_interface__["descr"]` which contains the full structured type information. This is inconsistent and breaks reasonable user expectations.

## Fix

```diff
--- a/numpy/ctypeslib/_ctypeslib.py
+++ b/numpy/ctypeslib/_ctypeslib.py
@@ -595,7 +595,12 @@ def as_ctypes(obj):
         if readonly:
             raise TypeError("readonly arrays unsupported")

-        ctype_scalar = as_ctypes_type(ai["typestr"])
+        # Use 'descr' for structured dtypes, 'typestr' for simple dtypes
+        if 'descr' in ai and len(ai['descr']) > 0 and ai['descr'][0][0]:
+            # Non-empty field name means structured dtype
+            ctype_scalar = as_ctypes_type(ai['descr'])
+        else:
+            ctype_scalar = as_ctypes_type(ai["typestr"])
         result_type = _ctype_ndarray(ctype_scalar, ai["shape"])
         result = result_type.from_address(addr)
         result.__keep = obj
```