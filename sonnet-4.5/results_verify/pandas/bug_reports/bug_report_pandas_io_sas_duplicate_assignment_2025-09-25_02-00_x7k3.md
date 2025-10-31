# Bug Report: pandas.io.sas Duplicate Assignment

**Target**: `pandas.io.sas.sas7bdat.SAS7BDATReader.__init__`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `SAS7BDATReader.__init__` method contains a duplicate assignment to `self._current_row_in_file_index` on consecutive lines, which is redundant and indicates a potential copy-paste error.

## Property-Based Test

This bug was discovered through code inspection rather than property-based testing, but the property violated is:

**Property**: Each instance variable should be initialized exactly once in `__init__` (idempotence of initialization).

## Reproducing the Bug

The bug exists in the source code itself and can be observed by reading the file:

```python
with open('/path/to/pandas/io/sas/sas7bdat.py') as f:
    lines = f.readlines()
    print(lines[204:208])
```

Expected output showing the duplicate:
```python
        self._current_row_in_file_index = 0
        self._current_row_on_page_index = 0
        self._current_row_in_file_index = 0  # Duplicate assignment
```

## Why This Is A Bug

While this bug does not cause functional issues (the second assignment simply overwrites the first with the same value), it represents:

1. **Code smell**: Duplicate code that serves no purpose
2. **Maintenance hazard**: If someone modifies one of these lines, they might miss the duplicate
3. **Possible intent error**: This may have been intended to initialize a different variable, such as `self._current_row_in_chunk_index` (which is used later but not initialized in `__init__`)

The pattern of variable names suggests this was likely a copy-paste error:
- `_current_row_in_file_index` (duplicated at lines 205 & 207)
- `_current_row_on_page_index` (line 206)
- `_current_row_in_chunk_index` (used but not initialized in `__init__`)

## Fix

Remove the duplicate assignment or replace it with initialization of the missing variable:

```diff
--- a/pandas/io/sas/sas7bdat.py
+++ b/pandas/io/sas/sas7bdat.py
@@ -204,7 +204,6 @@

         self._current_row_in_file_index = 0
         self._current_row_on_page_index = 0
-        self._current_row_in_file_index = 0

         self.handles = get_handle(
             path_or_buf, "rb", is_text=False, compression=compression
```

Or, if the intent was to initialize `_current_row_in_chunk_index`:

```diff
--- a/pandas/io/sas/sas7bdat.py
+++ b/pandas/io/sas/sas7bdat.py
@@ -204,7 +204,7 @@

         self._current_row_in_file_index = 0
         self._current_row_on_page_index = 0
-        self._current_row_in_file_index = 0
+        self._current_row_in_chunk_index = 0

         self.handles = get_handle(
             path_or_buf, "rb", is_text=False, compression=compression
```

Note: The second option requires verifying that `_current_row_in_chunk_index` is expected to be initialized to 0 in `__init__` rather than only when needed in the `read()` method (where it's currently set at line 685).