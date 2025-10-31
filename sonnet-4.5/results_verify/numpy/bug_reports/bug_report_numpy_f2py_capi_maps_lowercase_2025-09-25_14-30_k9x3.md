# Bug Report: numpy.f2py.capi_maps load_f2cmap_file Incorrect Lowercasing

**Target**: `numpy.f2py.capi_maps.load_f2cmap_file`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `load_f2cmap_file` function incorrectly lowercases the entire f2cmap file content, including C type names (values), when only Fortran type names and kind selectors (keys) should be lowercased. This breaks custom type mappings that use mixed-case C type names.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy.f2py.capi_maps as capi_maps
import tempfile
import os

@st.composite
def f2cmap_dict(draw):
    """Generate f2cmap dictionaries with mixed-case C type names."""
    # Fortran types (will be lowercased - OK)
    f_type = draw(st.sampled_from(['real', 'integer', 'complex']))
    # Kind selector (will be lowercased - OK)
    kind = draw(st.text(alphabet='abcdefghijklmnopqrstuvwxyz0123456789', min_size=1, max_size=10))
    # C type name (should preserve case - BUG!)
    c_type_base = draw(st.sampled_from(['float', 'double', 'int']))
    # Add some uppercase to make it mixed-case
    c_type = c_type_base.capitalize() + draw(st.text(alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ', max_size=5))

    return {f_type: {kind: c_type}}

@given(f2cmap_dict())
def test_load_f2cmap_preserves_c_type_case(f2cmap_content):
    """Test that load_f2cmap_file preserves case in C type names."""
    # Create temp file with f2cmap content
    with tempfile.NamedTemporaryFile(mode='w', suffix='.f2cmap', delete=False) as f:
        f.write(repr(f2cmap_content))
        fname = f.name

    try:
        # Load the file
        capi_maps.load_f2cmap_file(fname)

        # Check that C type names (values) preserved their case
        for f_type, kinds in f2cmap_content.items():
            for kind, c_type in kinds.items():
                # After loading, the mapping should preserve c_type case
                # But due to bug, c_type.lower() is stored instead
                expected_key = f_type.lower()  # Keys should be lowercase
                expected_subkey = kind.lower()  # Subkeys should be lowercase
                expected_value = c_type  # Values should preserve case!

                actual_value = capi_maps.f2cmap_all.get(expected_key, {}).get(expected_subkey)

                assert actual_value == expected_value, \
                    f"C type case not preserved: expected {expected_value}, got {actual_value}"
    finally:
        os.unlink(fname)
```

**Failing input**: `{'real': {'custom': 'MyCustomType'}}`

## Reproducing the Bug

```python
import numpy.f2py.capi_maps as capi_maps
import tempfile
import os

f2cmap_content = {'real': {'MyKind': 'MyCustomType'}}

with tempfile.NamedTemporaryFile(mode='w', suffix='.f2cmap', delete=False) as f:
    f.write(repr(f2cmap_content))
    fname = f.name

capi_maps.load_f2cmap_file(fname)

result = capi_maps.f2cmap_all.get('real', {}).get('mykind')

print(f"Original C type: {f2cmap_content['real']['MyKind']}")
print(f"After loading:   {result}")
print(f"Match: {result == 'MyCustomType'}")

os.unlink(fname)
```

**Output**:
```
Original C type: MyCustomType
After loading:   mycustomtype
Match: False
```

## Why This Is A Bug

**Root cause**: Line 159 in `capi_maps.py`:
```python
d = eval(f.read().lower(), {}, {})
```

This lowercases the **entire** file content, including C type names.

**Evidence from code**:

1. The `process_f2cmap_dict` function (called on line 160) **already** lowercases keys:
   ```python
   # From auxfuncs.py lines 978-981
   for k, d1 in new_map.items():
       d1_lower = {k1.lower(): v1 for k1, v1 in d1.items()}  # Only k1 lowercased!
       new_map_lower[k.lower()] = d1_lower
   ```

2. The docstring of `process_f2cmap_dict` says:
   > "It ensures that all **keys** are in lowercase"

   Not "all keys **and values**" - only keys!

3. Fortran is case-insensitive (keys should be lowercase) ✓
4. C is case-sensitive (values should preserve case) ✗

**Impact**:
- Breaks custom f2cmap files with mixed-case C type names
- Standard library types like `size_t`, `FILE`, `ptrdiff_t` would become `size_t`, `file`, `ptrdiff_t`
- User-defined types like `Real64`, `MyFloat`, `ComplexDouble` become `real64`, `myfloat`, `complexdouble`
- Silent corruption - no error message, just wrong type name used

## Fix

Remove the `.lower()` call from `load_f2cmap_file` since `process_f2cmap_dict` already handles key lowercasing correctly:

```diff
--- a/numpy/f2py/capi_maps.py
+++ b/numpy/f2py/capi_maps.py
@@ -156,7 +156,7 @@ def load_f2cmap_file(f2cmap_file):
     try:
         outmess(f'Reading f2cmap from {f2cmap_file!r} ...\n')
         with open(f2cmap_file) as f:
-            d = eval(f.read().lower(), {}, {})
+            d = eval(f.read(), {}, {})
         f2cmap_all, f2cmap_mapped = process_f2cmap_dict(f2cmap_all, d, c2py_map, True)
         outmess('Successfully applied user defined f2cmap changes\n')
     except Exception as msg:
```

This preserves C type names (values) while `process_f2cmap_dict` ensures Fortran types and kind selectors (keys) are lowercase.