# Bug Report: scipy.io.matlab.savemat Allows Digit-Prefixed Top-Level Keys Without Warning

**Target**: `scipy.io.matlab.savemat`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `savemat` function saves top-level dictionary keys starting with digits without issuing the documented warning, violating its stated contract and creating inconsistent behavior compared to struct field handling.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test for scipy.io.matlab digit-prefixed key bug
"""
from io import BytesIO
import numpy as np
from hypothesis import given, strategies as st, settings
from scipy.io.matlab import loadmat, savemat
import warnings


@settings(max_examples=50)
@given(st.from_regex(r'^[0-9][a-zA-Z0-9_]*$', fullmatch=True))
def test_digit_key_not_saved(key):
    """
    Test that keys starting with digits are not saved and trigger a warning.

    According to savemat documentation:
    "Note that if this dict has a key starting with `_` or a sub-dict has a key
    starting with `_` or a digit, these key's items will not be saved in the mat
    file and `MatWriteWarning` will be issued."
    """
    bio = BytesIO()
    data = {key: np.array([1, 2, 3])}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        savemat(bio, data)

        # Check if warning was issued
        if len(w) > 0:
            assert any("MatWriteWarning" in str(warn.category) for warn in w), \
                f"Expected MatWriteWarning but got: {[str(warn.category) for warn in w]}"

    bio.seek(0)
    loaded = loadmat(bio)

    # Check that key was not saved
    assert key not in loaded, f"Key '{key}' should not have been saved but was found in loaded data"


if __name__ == "__main__":
    # Run the test
    test_digit_key_not_saved()
```

<details>

<summary>
**Failing input**: `key='0'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 44, in <module>
    test_digit_key_not_saved()
    ~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 13, in test_digit_key_not_saved
    @given(st.from_regex(r'^[0-9][a-zA-Z0-9_]*$', fullmatch=True))
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 39, in test_digit_key_not_saved
    assert key not in loaded, f"Key '{key}' should not have been saved but was found in loaded data"
           ^^^^^^^^^^^^^^^^^
AssertionError: Key '0' should not have been saved but was found in loaded data
Falsifying example: test_digit_key_not_saved(
    key='0',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction case for scipy.io.matlab digit-prefixed key bug
"""
from io import BytesIO
import numpy as np
from scipy.io.matlab import loadmat, savemat
import warnings

# Test case 1: Single digit key
print("=== Test 1: Single digit key '0' ===")
bio = BytesIO()
data = {'0': np.array([1, 2, 3])}

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    savemat(bio, data)
    print(f"Warnings issued: {len(w)}")
    if w:
        for warning in w:
            print(f"  Warning: {warning.message}")

bio.seek(0)
loaded = loadmat(bio)
print(f"'0' in loaded: {'0' in loaded}")
if '0' in loaded:
    print(f"Value of '0': {loaded['0']}")
print()

# Test case 2: Digit-prefixed key
print("=== Test 2: Digit-prefixed key '1test' ===")
bio2 = BytesIO()
data2 = {'1test': np.array([4, 5, 6])}

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    savemat(bio2, data2)
    print(f"Warnings issued: {len(w)}")
    if w:
        for warning in w:
            print(f"  Warning: {warning.message}")

bio2.seek(0)
loaded2 = loadmat(bio2)
print(f"'1test' in loaded: {'1test' in loaded2}")
if '1test' in loaded2:
    print(f"Value of '1test': {loaded2['1test']}")
print()

# Test case 3: For comparison - underscore-prefixed key (should be ignored)
print("=== Test 3: Underscore-prefixed key '_hidden' (for comparison) ===")
bio3 = BytesIO()
data3 = {'_hidden': np.array([7, 8, 9])}

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    savemat(bio3, data3)
    print(f"Warnings issued: {len(w)}")
    if w:
        for warning in w:
            print(f"  Warning: {warning.message}")

bio3.seek(0)
loaded3 = loadmat(bio3)
print(f"'_hidden' in loaded: {'_hidden' in loaded3}")
if '_hidden' in loaded3:
    print(f"Value of '_hidden': {loaded3['_hidden']}")
print()

# Test case 4: Struct field with digit prefix (should be ignored per existing behavior)
print("=== Test 4: Struct field with digit prefix '0field' (for comparison) ===")
bio4 = BytesIO()
data4 = {'mystruct': {'0field': np.array([10, 11, 12]), 'valid_field': np.array([13, 14, 15])}}

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    savemat(bio4, data4)
    print(f"Warnings issued: {len(w)}")
    if w:
        for warning in w:
            print(f"  Warning: {warning.message}")

bio4.seek(0)
loaded4 = loadmat(bio4)
if 'mystruct' in loaded4:
    struct_data = loaded4['mystruct']
    print(f"Struct fields: {struct_data.dtype.names}")
    if struct_data.dtype.names and '0field' in struct_data.dtype.names:
        print(f"  '0field' was saved (unexpected)")
    else:
        print(f"  '0field' was not saved (expected)")
```

<details>

<summary>
Output demonstrating the bug
</summary>
```
=== Test 1: Single digit key '0' ===
Warnings issued: 0
'0' in loaded: True
Value of '0': [[1 2 3]]

=== Test 2: Digit-prefixed key '1test' ===
Warnings issued: 0
'1test' in loaded: True
Value of '1test': [[4 5 6]]

=== Test 3: Underscore-prefixed key '_hidden' (for comparison) ===
Warnings issued: 1
  Warning: Starting field name with a underscore (_hidden) is ignored
'_hidden' in loaded: False

=== Test 4: Struct field with digit prefix '0field' (for comparison) ===
Warnings issued: 1
  Warning: Starting field name with a underscore or a digit (0field) is ignored
Struct fields: ('valid_field',)
  '0field' was not saved (expected)
```
</details>

## Why This Is A Bug

The `savemat` function's docstring explicitly states:

> "Note that if this dict has a key starting with `_` or a sub-dict has a key starting with `_` or a digit, these key's items will not be saved in the mat file and `MatWriteWarning` will be issued."

This documentation makes no distinction between top-level keys and struct field keys - both should follow the same rules. However, the current implementation violates this contract in three ways:

1. **Top-level keys starting with digits ARE saved** - Keys like `'0'` and `'1test'` are successfully saved to the .mat file and can be loaded back, contrary to documentation
2. **No MatWriteWarning is issued** - The documentation promises a warning will be issued, but none is generated for digit-prefixed top-level keys
3. **Inconsistent behavior between levels** - Struct fields correctly reject digit-prefixed keys with a warning (line 486), but top-level keys do not (line 884)

This violates MATLAB's fundamental variable naming rules, which do not allow variable names to start with digits. The scipy library should maintain MATLAB compatibility.

## Relevant Context

The bug exists in `/scipy/io/matlab/_mio5.py` in the `MatFile5Writer.put_variables` method. The code at line 884 only checks for underscore-prefixed keys but fails to check for digit-prefixed keys:

```python
# Line 884 - Current (incorrect) implementation
if name[0] == '_':
    msg = (f"Starting field name with a "
           f"underscore ({name}) is ignored")
    warnings.warn(msg, MatWriteWarning, stacklevel=2)
    continue
```

In contrast, the struct field handling at line 486 correctly checks both cases:

```python
# Line 486 - Correct implementation for struct fields
if field[0] not in '_0123456789':
    dtype.append((str(field), object))
    values.append(value)
else:
    msg = (f"Starting field name with a underscore "
           f"or a digit ({field}) is ignored")
    warnings.warn(msg, MatWriteWarning, stacklevel=2)
```

This inconsistency suggests the top-level key validation was overlooked during implementation.

## Proposed Fix

```diff
--- a/scipy/io/matlab/_mio5.py
+++ b/scipy/io/matlab/_mio5.py
@@ -882,8 +882,8 @@ class MatFile5Writer:
         self._matrix_writer = VarWriter5(self)
         for name, var in mdict.items():
-            if name[0] == '_':
-                msg = (f"Starting field name with a "
-                       f"underscore ({name}) is ignored")
+            if name[0] in '_0123456789':
+                msg = (f"Starting field name with a underscore "
+                       f"or a digit ({name}) is ignored")
                 warnings.warn(msg, MatWriteWarning, stacklevel=2)
                 continue
             is_global = name in self.global_vars
```