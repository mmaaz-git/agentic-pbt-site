# Bug Report: ExcelWriter.check_extension Accepts Invalid Extensions

**Target**: `pandas.io.excel.ExcelWriter.check_extension`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `check_extension` method uses substring matching instead of exact matching to validate file extensions. This causes it to incorrectly accept invalid extensions like `.l`, `.x`, `.s`, `.m`, etc., because they are substrings of valid extensions like `.xlsx`, `.xlsm`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.io.excel._openpyxl import OpenpyxlWriter

@given(
    ext=st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=1, max_size=3).filter(
        lambda x: x not in ['xlsx', 'xlsm', 'xls', 'xlsb', 'ods', 'odt', 'odf']
    )
)
def test_check_extension_substring_bug(ext):
    try:
        result = OpenpyxlWriter.check_extension(f'.{ext}')
        supported = OpenpyxlWriter._supported_extensions
        is_substring_of_any = any(ext in extension for extension in supported)
        is_exact_match = any(f'.{ext}' == extension for extension in supported)

        if not is_exact_match and is_substring_of_any:
            raise AssertionError(
                f"Bug found: '.{ext}' was accepted but is not a supported extension. "
                f"It's a substring of {[e for e in supported if ext in e]} but not an exact match."
            )
    except ValueError:
        pass
```

**Failing input**: `ext='l'`

## Reproducing the Bug

```python
from pandas.io.excel._openpyxl import OpenpyxlWriter

print("Supported extensions:", OpenpyxlWriter._supported_extensions)

result = OpenpyxlWriter.check_extension('.l')
print(f"check_extension('.l') returned: {result}")

result = OpenpyxlWriter.check_extension('.x')
print(f"check_extension('.x') returned: {result}")

result = OpenpyxlWriter.check_extension('.s')
print(f"check_extension('.s') returned: {result}")

try:
    OpenpyxlWriter.check_extension('.pdf')
except ValueError as e:
    print(f"check_extension('.pdf') raised ValueError: {e}")
```

Output:
```
Supported extensions: ('.xlsx', '.xlsm')
check_extension('.l') returned: True
check_extension('.x') returned: True
check_extension('.s') returned: True
check_extension('.pdf') raised ValueError: Invalid extension for engine '...': 'pdf'
```

## Why This Is A Bug

The `check_extension` method is supposed to validate that a file extension is supported by the writer engine. However, it uses substring matching (`ext in extension`) instead of exact matching. This means:

- `.l` is incorrectly accepted because `'l' in '.xlsx'` and `'l' in '.xlsm'`
- `.x` is incorrectly accepted because `'x' in '.xlsx'` and `'x' in '.xlsm'`
- `.s` is incorrectly accepted because `'s' in '.xlsx'` and `'s' in '.xlsm'`

This violates the documented purpose of the method and could lead to confusing errors later when trying to write files with these invalid extensions.

## Fix

```diff
--- a/pandas/io/excel/_base.py
+++ b/pandas/io/excel/_base.py
@@ -410,7 +410,7 @@ class ExcelWriter(Generic[_WorkbookT]):
         """
         if ext.startswith("."):
             ext = ext[1:]
-        if not any(ext in extension for extension in cls._supported_extensions):
+        if not any(f'.{ext}' == extension for extension in cls._supported_extensions):
             raise ValueError(f"Invalid extension for engine '{cls.engine}': '{ext}'")
         return True
```