# Bug Report: numpy.f2py.capi_maps Incorrect complex_long_double Precision

**Target**: `numpy.f2py.capi_maps.c2capi_map['complex_long_double']`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `complex_long_double` type incorrectly maps to `NPY_CDOUBLE` instead of `NPY_CLONGDOUBLE`, causing precision loss when working with long double complex numbers.

## Property-Based Test

```python
from numpy.f2py import capi_maps

def test_complex_types_preserve_precision():
    base_types = ['float', 'double', 'long_double']

    for base_type in base_types:
        if base_type not in capi_maps.c2capi_map:
            continue

        complex_type = f'complex_{base_type}'
        if complex_type not in capi_maps.c2capi_map:
            continue

        base_npy = capi_maps.c2capi_map[base_type]
        complex_npy = capi_maps.c2capi_map[complex_type]

        base_precision = base_npy.replace('NPY_', '')
        complex_precision = complex_npy.replace('NPY_C', '')

        assert base_precision.upper() == complex_precision.upper(), \
            f"Precision mismatch: {base_type} -> {base_npy}, but {complex_type} -> {complex_npy}"
```

**Failing input**: `base_type='long_double'`

## Reproducing the Bug

```python
from numpy.f2py import capi_maps

print("float precision:")
print(f"  float: {capi_maps.c2capi_map['float']}")
print(f"  complex_float: {capi_maps.c2capi_map['complex_float']}")

print("\ndouble precision:")
print(f"  double: {capi_maps.c2capi_map['double']}")
print(f"  complex_double: {capi_maps.c2capi_map['complex_double']}")

print("\nlong_double precision:")
print(f"  long_double: {capi_maps.c2capi_map['long_double']}")
print(f"  complex_long_double: {capi_maps.c2capi_map['complex_long_double']}")

print("\nBUG: complex_long_double should map to NPY_CLONGDOUBLE, not NPY_CDOUBLE")
```

## Why This Is A Bug

The pattern across all other numeric types shows that complex types should use the same precision as their base type:
- `float` (32-bit) → `NPY_FLOAT`, `complex_float` → `NPY_CFLOAT`
- `double` (64-bit) → `NPY_DOUBLE`, `complex_double` → `NPY_CDOUBLE`
- `long_double` (≥64-bit) → `NPY_LONGDOUBLE`, `complex_long_double` → `NPY_CDOUBLE` ✗

The mapping for `complex_long_double` at line 73 should be `NPY_CLONGDOUBLE` to preserve long double precision. Using `NPY_CDOUBLE` causes silent precision loss when users work with extended-precision complex numbers.

## Fix

```diff
--- a/numpy/f2py/capi_maps.py
+++ b/numpy/f2py/capi_maps.py
@@ -70,7 +70,7 @@ c2capi_map = {'double': 'NPY_DOUBLE',
                 'unsigned_long_long': 'NPY_ULONGLONG',
                 'complex_float': 'NPY_CFLOAT',
                 'complex_double': 'NPY_CDOUBLE',
-                'complex_long_double': 'NPY_CDOUBLE',
+                'complex_long_double': 'NPY_CLONGDOUBLE',
                 'string': 'NPY_STRING',
                 'character': 'NPY_STRING'}
```