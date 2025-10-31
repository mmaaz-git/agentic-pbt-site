# Bug Report: scipy.io.matlab.savemat - No Warning for Variable Names Starting with Digits

**Target**: `scipy.io.matlab.savemat`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`savemat` fails to issue a `MatWriteWarning` for dictionary keys starting with digits, and incorrectly saves them to the file. This violates the documented behavior and MATLAB's variable naming rules, potentially creating files that cannot be loaded in MATLAB.

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
        st.text(min_size=1, max_size=10).filter(lambda s: s[0] == '_'),
        st.text(min_size=1, max_size=10).filter(lambda s: s[0].isdigit()),
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
```

**Failing input**: `invalid_key='0'` (or any string starting with a digit)

## Reproducing the Bug

```python
import io
import warnings
from scipy.io import savemat, loadmat
from scipy.io.matlab import MatWriteWarning

file_obj = io.BytesIO()
data = {'0test': 123, 'validname': 456}

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    savemat(file_obj, data)

    mat_warnings = [warning for warning in w if issubclass(warning.category, MatWriteWarning)]
    print(f"MatWriteWarnings issued: {len(mat_warnings)}")

file_obj.seek(0)
loaded = loadmat(file_obj)
user_keys = [k for k in loaded.keys() if not k.startswith('__')]
print(f"Keys in saved file: {user_keys}")
```

Output:
```
MatWriteWarnings issued: 0
Keys in saved file: ['0test', 'validname']
```

## Why This Is A Bug

The `savemat` docstring explicitly states:

> "Note that if this dict has a key starting with `_` or a sub-dict has a key starting with `_` **or a digit**, these key's items will not be saved in the mat file and `MatWriteWarning` will be issued."

The actual behavior contradicts this in two ways:

1. **No warning is issued** for keys starting with digits
2. **The data IS saved** despite documentation saying it won't be

Additionally, MATLAB itself does not allow variable names starting with digits. Saving such keys creates .mat files that violate MATLAB's naming conventions and may not be loadable in MATLAB, causing data integrity issues.

Keys starting with `_` work correctly (warning issued, data not saved), but keys starting with digits do not.

## Fix

The validation logic for keys starting with digits should be implemented similarly to the existing validation for keys starting with underscores. Based on the behavior observed, the check likely exists in the mat file writer but only validates underscore prefixes.

The fix should:
1. Add validation to detect keys starting with digits
2. Issue `MatWriteWarning` when such keys are found
3. Skip saving those keys to the file (as documented)

Without access to the exact implementation, the fix would be in `/scipy/io/matlab/_mio5.py` or similar, where field name validation occurs. The existing underscore check should be extended to also check `key[0].isdigit()`.