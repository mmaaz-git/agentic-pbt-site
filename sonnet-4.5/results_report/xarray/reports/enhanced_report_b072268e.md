# Bug Report: xarray.backends.netcdf3.is_valid_nc3_name IndexError on Empty String

**Target**: `xarray.backends.netcdf3.is_valid_nc3_name`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `is_valid_nc3_name()` validation function crashes with an `IndexError` when passed an empty string, instead of returning `False` as expected for invalid input.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from xarray.backends.netcdf3 import is_valid_nc3_name

@given(st.text())
def test_is_valid_nc3_name_doesnt_crash(s):
    result = is_valid_nc3_name(s)
    assert isinstance(result, bool)

# Run the test
test_is_valid_nc3_name_doesnt_crash()
```

<details>

<summary>
**Failing input**: `s=''`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/31/hypo.py", line 10, in <module>
    test_is_valid_nc3_name_doesnt_crash()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/31/hypo.py", line 5, in test_is_valid_nc3_name_doesnt_crash
    def test_is_valid_nc3_name_doesnt_crash(s):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/31/hypo.py", line 6, in test_is_valid_nc3_name_doesnt_crash
    result = is_valid_nc3_name(s)
  File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/backends/netcdf3.py", line 167, in is_valid_nc3_name
    and (s[-1] != " ")
         ~^^^^
IndexError: string index out of range
Falsifying example: test_is_valid_nc3_name_doesnt_crash(
    s='',
)
```
</details>

## Reproducing the Bug

```python
from xarray.backends.netcdf3 import is_valid_nc3_name

# Test empty string which should return False but crashes instead
result = is_valid_nc3_name("")
print(f"Result: {result}")
```

<details>

<summary>
IndexError: string index out of range
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/31/repo.py", line 4, in <module>
    result = is_valid_nc3_name("")
  File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/backends/netcdf3.py", line 167, in is_valid_nc3_name
    and (s[-1] != " ")
         ~^^^^
IndexError: string index out of range
```
</details>

## Why This Is A Bug

This violates expected behavior for a validation function. According to the function's docstring, `is_valid_nc3_name()` should "test whether an object can be validly converted to a netCDF-3 dimension, variable or attribute name". As a validation function, it should return a boolean result (`True` for valid, `False` for invalid) for all possible inputs, not crash with an uncaught exception.

The function already demonstrates proper handling of invalid inputs at line 159-160 where it checks `if not isinstance(s, str): return False`, showing clear intent to return `False` for invalid inputs rather than raising exceptions. An empty string is clearly an invalid NetCDF-3 name (the NetCDF-3 specification requires names to match a regex pattern that mandates at least one character), so the function should return `False`.

The crash occurs due to unchecked array access at two locations:
- Line 167: `s[-1] != " "` - attempts to access the last character without verifying the string is non-empty
- Line 168: `s[0]` - attempts to access the first character without bounds checking

This makes the validation function unreliable for validating untrusted input, defeating its primary purpose as a gatekeeper for data integrity.

## Relevant Context

The NetCDF-3 file format specification defines valid names using the regex pattern:
`([a-zA-Z0-9_]|{MUTF8})([^\x00-\x1F/\x7F-\xFF]|{MUTF8})*`

This pattern requires at least one character (the first character group is mandatory), making empty strings invalid by specification.

The function is part of xarray's NetCDF-3 backend implementation and is used to validate names before writing to NetCDF-3 files. The validation ensures compatibility with the NetCDF-3 format restrictions.

Source code location: `/home/npc/miniconda/lib/python3.13/site-packages/xarray/backends/netcdf3.py:142-171`

## Proposed Fix

```diff
--- a/xarray/backends/netcdf3.py
+++ b/xarray/backends/netcdf3.py
@@ -159,6 +159,8 @@ def is_valid_nc3_name(s):
     if not isinstance(s, str):
         return False
+    if len(s) == 0:
+        return False
     num_bytes = len(s.encode("utf-8"))
     return (
         (unicodedata.normalize("NFC", s) == s)
```