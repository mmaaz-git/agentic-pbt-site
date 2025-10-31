# Bug Report: xarray.backends.netcdf3.is_valid_nc3_name IndexError on Empty String

**Target**: `xarray.backends.netcdf3.is_valid_nc3_name`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `is_valid_nc3_name` function crashes with an `IndexError` when given an empty string input, instead of returning `False` as expected for an invalid netCDF-3 name.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from xarray.backends.netcdf3 import is_valid_nc3_name

@given(st.text())
@settings(max_examples=1000)
def test_is_valid_nc3_name_does_not_crash(name):
    result = is_valid_nc3_name(name)
    assert isinstance(result, bool)

# Run the test
if __name__ == "__main__":
    test_is_valid_nc3_name_does_not_crash()
```

<details>

<summary>
**Failing input**: `''`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 12, in <module>
    test_is_valid_nc3_name_does_not_crash()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 5, in test_is_valid_nc3_name_does_not_crash
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 7, in test_is_valid_nc3_name_does_not_crash
    result = is_valid_nc3_name(name)
  File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/backends/netcdf3.py", line 167, in is_valid_nc3_name
    and (s[-1] != " ")
         ~^^^^
IndexError: string index out of range
Falsifying example: test_is_valid_nc3_name_does_not_crash(
    name='',
)
```
</details>

## Reproducing the Bug

```python
from xarray.backends.netcdf3 import is_valid_nc3_name

# Test with empty string to reproduce the bug
result = is_valid_nc3_name("")
print(f"Result: {result}")
```

<details>

<summary>
IndexError: string index out of range
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/3/repo.py", line 4, in <module>
    result = is_valid_nc3_name("")
  File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/backends/netcdf3.py", line 167, in is_valid_nc3_name
    and (s[-1] != " ")
         ~^^^^
IndexError: string index out of range
```
</details>

## Why This Is A Bug

The function `is_valid_nc3_name` is documented as a validation function that tests whether an object can be validly converted to a netCDF-3 dimension, variable, or attribute name. According to its docstring and expected behavior:

1. **Contract Violation**: The function should return a boolean value (`True` or `False`) for all string inputs. Instead, it crashes with an unhandled `IndexError` when given an empty string.

2. **Inconsistent Error Handling**: The function correctly returns `False` for non-string inputs (line 159-160), demonstrating it's designed to handle invalid inputs gracefully. However, it fails to handle empty strings, which are clearly invalid netCDF-3 names.

3. **Specification Violation**: According to the netCDF-3 specification mentioned in the docstring, valid names must start with an alphanumeric character, a multi-byte UTF-8 character, or underscore. An empty string has no characters and therefore cannot be a valid name.

4. **Implementation Error**: The bug occurs at line 167 where the code checks `s[-1] != " "` to ensure names don't have trailing spaces. This check doesn't verify that the string is non-empty before attempting to access the last character. Similarly, line 168 tries to access `s[0]` which would also fail on an empty string.

5. **Logic Error**: Line 165 checks `num_bytes >= 0`, which is always true since `len()` always returns non-negative values. This should be `num_bytes > 0` to reject empty strings.

## Relevant Context

The `is_valid_nc3_name` function is part of xarray's netCDF-3 backend implementation, used for validating names when writing netCDF-3 files. The netCDF-3 format has strict naming requirements as outlined in the function's docstring.

The function is located in `/xarray/backends/netcdf3.py` at line 142-170. It performs multiple validation checks including:
- Unicode normalization check
- Reserved name check
- Invalid character checks (`/` not allowed)
- No trailing spaces
- First character must be alphanumeric, multi-byte UTF-8, or underscore
- Subsequent characters validation

The issue was discovered through property-based testing with Hypothesis, which systematically tests edge cases including empty strings - a common boundary condition that should be handled in validation functions.

## Proposed Fix

```diff
--- a/xarray/backends/netcdf3.py
+++ b/xarray/backends/netcdf3.py
@@ -159,11 +159,12 @@ def is_valid_nc3_name(s):
     if not isinstance(s, str):
         return False
     num_bytes = len(s.encode("utf-8"))
+    if num_bytes == 0:
+        return False
     return (
         (unicodedata.normalize("NFC", s) == s)
         and (s not in _reserved_names)
-        and (num_bytes >= 0)
         and ("/" not in s)
         and (s[-1] != " ")
         and (_isalnumMUTF8(s[0]) or (s[0] == "_"))
         and all(_isalnumMUTF8(c) or c in _specialchars for c in s)
     )
```