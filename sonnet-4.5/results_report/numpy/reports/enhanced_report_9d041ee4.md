# Bug Report: numpy.ctypeslib.as_ctypes Fails on Structured Arrays

**Target**: `numpy.ctypeslib.as_ctypes`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`numpy.ctypeslib.as_ctypes` raises NotImplementedError when converting numpy arrays with structured dtypes to ctypes objects, despite the underlying infrastructure fully supporting this conversion.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings


@given(
    st.lists(
        st.tuples(
            st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122),
                    min_size=1, max_size=10),
            st.sampled_from([np.int32, np.float64])
        ),
        min_size=1,
        max_size=5,
        unique_by=lambda x: x[0]
    )
)
@settings(max_examples=200)
def test_structured_array_conversion(field_specs):
    """Test that numpy arrays with structured dtypes can be converted to ctypes objects."""
    dtype = np.dtype([(name, dt) for name, dt in field_specs])
    arr = np.zeros(10, dtype=dtype)

    # This should work but currently raises NotImplementedError
    ctypes_obj = np.ctypeslib.as_ctypes(arr)
    result = np.ctypeslib.as_array(ctypes_obj)

    # Verify round-trip conversion preserves data
    for name, _ in field_specs:
        np.testing.assert_array_equal(result[name], arr[name])


if __name__ == "__main__":
    # Run the test to find a failing case
    test_structured_array_conversion()
```

<details>

<summary>
**Failing input**: `field_specs=[('a', numpy.int32)]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 34, in <module>
    test_structured_array_conversion()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 6, in test_structured_array_conversion
    st.lists(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 24, in test_structured_array_conversion
    ctypes_obj = np.ctypeslib.as_ctypes(arr)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/ctypeslib/_ctypeslib.py", line 599, in as_ctypes
    ctype_scalar = as_ctypes_type(ai["typestr"])
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/ctypeslib/_ctypeslib.py", line 518, in as_ctypes_type
    return _ctype_from_dtype(np.dtype(dtype))
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/ctypeslib/_ctypeslib.py", line 461, in _ctype_from_dtype
    return _ctype_from_dtype_scalar(dtype)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/ctypeslib/_ctypeslib.py", line 387, in _ctype_from_dtype_scalar
    raise NotImplementedError(
        f"Converting {dtype!r} to a ctypes type"
    ) from None
NotImplementedError: Converting dtype('V4') to a ctypes type
Falsifying example: test_structured_array_conversion(
    field_specs=[('a', numpy.int32)],  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import numpy as np

# Create a structured dtype with two fields
dtype = np.dtype([('x', np.int32), ('y', np.float64)])

# Create an array with this structured dtype
arr = np.array([(1, 1.5), (2, 2.5), (3, 3.5)], dtype=dtype)

print("=" * 60)
print("Testing numpy.ctypeslib.as_ctypes with structured array")
print("=" * 60)
print(f"\nArray created:")
print(f"  data: {arr}")
print(f"  dtype: {arr.dtype}")
print(f"  shape: {arr.shape}")

# Show what the array interface typestr looks like for structured arrays
print(f"\nArray interface info:")
ai = arr.__array_interface__
print(f"  typestr: {ai['typestr']}")
print(f"  shape: {ai['shape']}")

print("\n" + "-" * 60)
print("Attempting np.ctypeslib.as_ctypes(arr):")
print("-" * 60)
try:
    ctypes_obj = np.ctypeslib.as_ctypes(arr)
    print(f"Success! Result: {ctypes_obj}")
    print(f"Type: {type(ctypes_obj)}")
except NotImplementedError as e:
    print(f"ERROR - NotImplementedError: {e}")

    print("\n" + "-" * 60)
    print("However, as_ctypes_type CAN handle the dtype directly:")
    print("-" * 60)
    try:
        ctype = np.ctypeslib.as_ctypes_type(arr.dtype)
        print(f"  np.ctypeslib.as_ctypes_type(arr.dtype) = {ctype}")
        print(f"  This is a ctypes.Structure with the correct fields")
    except Exception as e2:
        print(f"  Error: {e2}")

    print("\n" + "-" * 60)
    print("The problem: as_ctypes passes typestr instead of dtype")
    print("-" * 60)
    print(f"  as_ctypes calls: as_ctypes_type(ai['typestr'])")
    print(f"  ai['typestr'] = '{ai['typestr']}' (void type, loses field info)")
    print(f"  Should call: as_ctypes_type(obj.dtype)")
    print(f"  obj.dtype = {arr.dtype} (preserves field info)")
```

<details>

<summary>
NotImplementedError when calling as_ctypes on structured array
</summary>
```
============================================================
Testing numpy.ctypeslib.as_ctypes with structured array
============================================================

Array created:
  data: [(1, 1.5) (2, 2.5) (3, 3.5)]
  dtype: [('x', '<i4'), ('y', '<f8')]
  shape: (3,)

Array interface info:
  typestr: |V12
  shape: (3,)

------------------------------------------------------------
Attempting np.ctypeslib.as_ctypes(arr):
------------------------------------------------------------
ERROR - NotImplementedError: Converting dtype('V12') to a ctypes type

------------------------------------------------------------
However, as_ctypes_type CAN handle the dtype directly:
------------------------------------------------------------
  np.ctypeslib.as_ctypes_type(arr.dtype) = <class 'struct'>
  This is a ctypes.Structure with the correct fields

------------------------------------------------------------
The problem: as_ctypes passes typestr instead of dtype
------------------------------------------------------------
  as_ctypes calls: as_ctypes_type(ai['typestr'])
  ai['typestr'] = '|V12' (void type, loses field info)
  Should call: as_ctypes_type(obj.dtype)
  obj.dtype = [('x', '<i4'), ('y', '<f8')] (preserves field info)
```
</details>

## Why This Is A Bug

This violates expected behavior in several critical ways:

1. **API Inconsistency**: The function `as_ctypes_type` successfully handles structured dtypes and is documented with examples showing this capability. Users reasonably expect `as_ctypes` to work with the same types that `as_ctypes_type` supports.

2. **Implementation Error**: The bug is a clear coding mistake on line 599 of `_ctypeslib.py`. The function incorrectly passes `ai["typestr"]` to `as_ctypes_type` instead of `obj.dtype`. For structured arrays, the typestr is a void type descriptor (e.g., '|V12') that contains only the total byte size but loses all field names and types. The dtype object contains the complete field specifications needed for proper conversion.

3. **Core Feature Broken**: Structured arrays are NumPy's standard mechanism for representing C structs. The ctypeslib module exists specifically to facilitate interfacing with C libraries. The inability to convert structured arrays to ctypes severely undermines the module's primary purpose, as C structs are fundamental to most C APIs.

4. **Documentation Mismatch**: The module documentation and docstrings never mention any limitation regarding structured arrays. The `as_ctypes` docstring states it creates "a ctypes object from a numpy array" without any dtype restrictions.

## Relevant Context

The ctypeslib module provides critical functionality for interfacing NumPy with C libraries through ctypes. The infrastructure for handling structured arrays already exists in the codebase:

- `_ctype_from_dtype_structured` (line 403-453) properly converts structured dtypes to ctypes.Structure
- `as_ctypes_type` (line 464-518) correctly delegates to this function when given a structured dtype
- The bug only affects `as_ctypes` due to the incorrect parameter being passed

Key code locations:
- Bug location: `/numpy/ctypeslib/_ctypeslib.py:599`
- Working structured dtype handler: `/numpy/ctypeslib/_ctypeslib.py:403-453`
- Documentation: https://numpy.org/doc/stable/reference/routines.ctypeslib.html

The `__array_interface__` protocol's typestr field uses NumPy's array protocol type strings, which for structured arrays become void types ('|Vn' where n is the byte size). This representation loses the field information essential for ctypes conversion.

## Proposed Fix

```diff
--- a/numpy/ctypeslib/_ctypeslib.py
+++ b/numpy/ctypeslib/_ctypeslib.py
@@ -596,7 +596,7 @@ def as_ctypes(obj):

         # can't use `_dtype((ai["typestr"], ai["shape"]))` here, as it overflows
         # dtype.itemsize (gh-14214)
-        ctype_scalar = as_ctypes_type(ai["typestr"])
+        ctype_scalar = as_ctypes_type(obj.dtype)
         result_type = _ctype_ndarray(ctype_scalar, ai["shape"])
         result = result_type.from_address(addr)
         result.__keep = obj
```