# Bug Report: scipy.io.matlab.savemat Fails to Validate Variable Names Starting with Digits

**Target**: `scipy.io.matlab.savemat`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`scipy.io.matlab.savemat` fails to issue `MatWriteWarning` for dictionary keys starting with digits and incorrectly saves them to the file, violating both the documented behavior and MATLAB's variable naming rules.

## Property-Based Test

```python
import io
import warnings
from hypothesis import given, strategies as st, settings
from scipy.io import savemat
from scipy.io.matlab import MatWriteWarning


@settings(max_examples=100)
@given(
    invalid_key=st.one_of(
        st.text(alphabet=st.characters(min_codepoint=33, max_codepoint=126), min_size=1, max_size=10).filter(lambda s: s[0] == '_'),
        st.text(alphabet=st.characters(min_codepoint=33, max_codepoint=126), min_size=1, max_size=10).filter(lambda s: s[0].isdigit()),
    ),
    value=st.integers(min_value=0, max_value=100)
)
def test_savemat_invalid_key_warning(invalid_key, value):
    file_obj = io.BytesIO()
    data = {invalid_key: value}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        savemat(file_obj, data)

        assert len(w) > 0, f"No warning issued for invalid key {invalid_key}"
        assert any(issubclass(warning.category, MatWriteWarning) for warning in w), \
            f"Expected MatWriteWarning for invalid key {invalid_key}"


if __name__ == "__main__":
    # Run the test
    test_savemat_invalid_key_warning()
```

<details>

<summary>
**Failing input**: `invalid_key='0', value=0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 31, in <module>
    test_savemat_invalid_key_warning()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 9, in test_savemat_invalid_key_warning
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 24, in test_savemat_invalid_key_warning
    assert len(w) > 0, f"No warning issued for invalid key {invalid_key}"
           ^^^^^^^^^^
AssertionError: No warning issued for invalid key 0
Falsifying example: test_savemat_invalid_key_warning(
    invalid_key='0',
    value=0,  # or any other generated value
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/.local/lib/python3.13/site-packages/scipy/io/matlab/_mio5.py:889
```
</details>

## Reproducing the Bug

```python
import io
import warnings
from scipy.io import savemat, loadmat
from scipy.io.matlab import MatWriteWarning

# Test case: keys starting with digits should trigger warning and not be saved
file_obj = io.BytesIO()
data = {'0test': 123, '9abc': 456, 'validname': 789}

print("Testing keys starting with digits...")
print(f"Input data: {data}")
print()

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    savemat(file_obj, data)

    mat_warnings = [warning for warning in w if issubclass(warning.category, MatWriteWarning)]
    print(f"Number of MatWriteWarnings issued: {len(mat_warnings)}")
    if mat_warnings:
        for warning in mat_warnings:
            print(f"  Warning message: {warning.message}")

# Load the saved data to see what was actually saved
file_obj.seek(0)
loaded = loadmat(file_obj)
user_keys = [k for k in loaded.keys() if not k.startswith('__')]
print(f"Keys actually saved in file: {user_keys}")
loaded_data = {k: loaded[k].item() if hasattr(loaded[k], 'item') else loaded[k] for k in user_keys}
print(f"Loaded data (excluding metadata): {loaded_data}")
print()

# Compare with underscore-prefixed keys (which should work correctly)
print("Testing keys starting with underscore for comparison...")
file_obj2 = io.BytesIO()
data2 = {'_test': 123, 'validname': 456}
print(f"Input data: {data2}")

with warnings.catch_warnings(record=True) as w2:
    warnings.simplefilter("always")
    savemat(file_obj2, data2)

    mat_warnings2 = [warning for warning in w2 if issubclass(warning.category, MatWriteWarning)]
    print(f"Number of MatWriteWarnings issued: {len(mat_warnings2)}")
    if mat_warnings2:
        for warning in mat_warnings2:
            print(f"  Warning message: {warning.message}")

file_obj2.seek(0)
loaded2 = loadmat(file_obj2)
user_keys2 = [k for k in loaded2.keys() if not k.startswith('__')]
print(f"Keys actually saved in file: {user_keys2}")
loaded_data2 = {k: loaded2[k].item() if hasattr(loaded2[k], 'item') else loaded2[k] for k in user_keys2}
print(f"Loaded data (excluding metadata): {loaded_data2}")
```

<details>

<summary>
Output demonstrating the bug
</summary>
```
Testing keys starting with digits...
Input data: {'0test': 123, '9abc': 456, 'validname': 789}

Number of MatWriteWarnings issued: 0
Keys actually saved in file: ['0test', '9abc', 'validname']
Loaded data (excluding metadata): {'0test': 123, '9abc': 456, 'validname': 789}

Testing keys starting with underscore for comparison...
Input data: {'_test': 123, 'validname': 456}
Number of MatWriteWarnings issued: 1
  Warning message: Starting field name with a underscore (_test) is ignored
Keys actually saved in file: ['validname']
Loaded data (excluding metadata): {'validname': 456}
```
</details>

## Why This Is A Bug

The `savemat` function's docstring explicitly states:

> "Note that if this dict has a key starting with `_` or a sub-dict has a key starting with `_` **or a digit**, these key's items will not be saved in the mat file and `MatWriteWarning` will be issued."

The actual behavior contradicts this documentation in two critical ways:

1. **No warning is issued** for keys starting with digits (violates the promise to issue `MatWriteWarning`)
2. **The data IS saved** to the file despite documentation stating it won't be

This creates multiple problems:
- **API Contract Violation**: The function doesn't behave as documented, breaking user expectations
- **MATLAB Incompatibility**: MATLAB does not allow variable names starting with digits. Files saved with such keys violate MATLAB's naming conventions and may fail to load in MATLAB
- **Inconsistent Behavior**: Keys starting with underscore work correctly (warning issued, data not saved), but keys starting with digits do not, creating an inconsistent user experience

## Relevant Context

The bug occurs in `/home/npc/.local/lib/python3.13/site-packages/scipy/io/matlab/_mio5.py` at the `put_variables` method of the `MatFile5Writer` class (lines 884-888). The code correctly validates and warns for keys starting with underscore but completely lacks validation for keys starting with digits at the top-level dictionary.

Interestingly, the validation for digit-prefixed keys DOES exist for sub-dictionaries/structures (line 486), but is missing for the top-level dictionary entries.

MATLAB's official documentation confirms that variable names cannot start with digits: https://www.mathworks.com/help/matlab/matlab_prog/variable-names.html

## Proposed Fix

The fix is straightforward - add validation for digit-prefixed keys in the `put_variables` method, similar to the existing underscore validation:

```diff
--- a/scipy/io/matlab/_mio5.py
+++ b/scipy/io/matlab/_mio5.py
@@ -881,7 +881,7 @@ class MatFile5Writer(MatFileWriter):
             self.write_file_header()
         self._matrix_writer = VarWriter5(self)
         for name, var in mdict.items():
-            if name[0] == '_':
+            if name[0] in '_0123456789':
                 msg = (f"Starting field name with a "
-                       f"underscore ({name}) is ignored")
+                       f"underscore or a digit ({name}) is ignored")
                 warnings.warn(msg, MatWriteWarning, stacklevel=2)
                 continue
             is_global = name in self.global_vars
```