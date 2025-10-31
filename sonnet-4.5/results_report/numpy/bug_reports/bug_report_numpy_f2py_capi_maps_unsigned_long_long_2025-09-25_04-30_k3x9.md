# Bug Report: numpy.f2py.capi_maps Missing unsigned_long_long Key

**Target**: `numpy.f2py.capi_maps`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `c2py_map` dictionary is missing the `'unsigned_long_long'` key despite it being present in `c2capi_map` and `c2pycode_map`, causing inconsistency across related mapping dictionaries.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from numpy.f2py import capi_maps

@given(st.sampled_from(list(capi_maps.c2capi_map.keys())))
def test_c2capi_keys_have_c2py_mapping(ctype):
    assert ctype in capi_maps.c2py_map, \
        f"C type {ctype!r} in c2capi_map but not in c2py_map"
```

**Failing input**: `'unsigned_long_long'`

## Reproducing the Bug

```python
from numpy.f2py import capi_maps

print(f"'unsigned_long_long' in c2capi_map: {'unsigned_long_long' in capi_maps.c2capi_map}")
print(f"'unsigned_long_long' in c2py_map: {'unsigned_long_long' in capi_maps.c2py_map}")

try:
    numpy_dtype = capi_maps.c2capi_map['unsigned_long_long']
    python_type = capi_maps.c2py_map['unsigned_long_long']
    print(f"Success: {numpy_dtype} -> {python_type}")
except KeyError as e:
    print(f"KeyError: {e}")
```

## Why This Is A Bug

The mapping dictionaries (`c2py_map`, `c2capi_map`, `c2pycode_map`) should have consistent keys since they represent different aspects of the same C types. The presence of `unsigned_long_long` in `c2capi_map` (line 70) and `c2pycode_map` (line 90) but its absence from `c2py_map` creates an inconsistency. Any code that expects to look up a type from one map in another will fail for `unsigned_long_long`.

## Fix

```diff
--- a/numpy/f2py/capi_maps.py
+++ b/numpy/f2py/capi_maps.py
@@ -46,6 +46,7 @@ c2py_map = {'double': 'float',
             'int': 'int',                              # forced casting
             'long': 'int',
             'long_long': 'long',
+            'unsigned_long': 'long',                   # forced casting
+            'unsigned_long_long': 'long',
             'unsigned': 'int',                         # forced casting
             'complex_float': 'complex',                # forced casting
             'complex_double': 'complex',
```