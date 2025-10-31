# Bug Report: pandas.io.clipboards Exception Type Inconsistency for Invalid Encodings

**Target**: `pandas.io.clipboards.read_clipboard()` and `pandas.io.clipboards.to_clipboard()`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `read_clipboard()` and `to_clipboard()` functions raise different exception types when rejecting the same invalid parameter value (non-UTF-8 encoding), creating an inconsistent API that complicates error handling.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
from pandas.io.clipboards import read_clipboard, to_clipboard


@given(st.sampled_from(['latin-1', 'iso-8859-1', 'cp1252', 'utf-16', 'ascii']))
@settings(max_examples=10)
def test_both_functions_reject_non_utf8(encoding):
    if encoding.lower().replace('-', '') == 'utf8':
        return

    read_exc_type = None
    write_exc_type = None

    try:
        read_clipboard(encoding=encoding)
    except Exception as e:
        read_exc_type = type(e).__name__

    try:
        to_clipboard(pd.DataFrame([[1, 2]]), encoding=encoding)
    except Exception as e:
        write_exc_type = type(e).__name__

    assert read_exc_type == write_exc_type, (
        f"Inconsistent exception types for encoding '{encoding}': "
        f"read_clipboard raises {read_exc_type}, to_clipboard raises {write_exc_type}"
    )

if __name__ == "__main__":
    test_both_functions_reject_non_utf8()
```

<details>

<summary>
**Failing input**: `encoding='latin-1'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 31, in <module>
    test_both_functions_reject_non_utf8()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 7, in test_both_functions_reject_non_utf8
    @settings(max_examples=10)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 25, in test_both_functions_reject_non_utf8
    assert read_exc_type == write_exc_type, (
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Inconsistent exception types for encoding 'latin-1': read_clipboard raises NotImplementedError, to_clipboard raises ValueError
Falsifying example: test_both_functions_reject_non_utf8(
    encoding='latin-1',
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
from pandas.io.clipboards import read_clipboard, to_clipboard

# Test read_clipboard with non-UTF-8 encoding
try:
    read_clipboard(encoding='latin-1')
except Exception as e:
    print(f"read_clipboard with 'latin-1': {type(e).__name__}: {e}")

# Test to_clipboard with non-UTF-8 encoding
try:
    df = pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
    to_clipboard(df, encoding='latin-1')
except Exception as e:
    print(f"to_clipboard with 'latin-1': {type(e).__name__}: {e}")

# Test with another non-UTF-8 encoding
try:
    read_clipboard(encoding='iso-8859-1')
except Exception as e:
    print(f"read_clipboard with 'iso-8859-1': {type(e).__name__}: {e}")

try:
    df = pd.DataFrame([[5, 6]], columns=['X', 'Y'])
    to_clipboard(df, encoding='iso-8859-1')
except Exception as e:
    print(f"to_clipboard with 'iso-8859-1': {type(e).__name__}: {e}")
```

<details>

<summary>
Exception type inconsistency confirmed across multiple encodings
</summary>
```
read_clipboard with 'latin-1': NotImplementedError: reading from clipboard only supports utf-8 encoding
to_clipboard with 'latin-1': ValueError: clipboard only supports utf-8 encoding
read_clipboard with 'iso-8859-1': NotImplementedError: reading from clipboard only supports utf-8 encoding
to_clipboard with 'iso-8859-1': ValueError: clipboard only supports utf-8 encoding
```
</details>

## Why This Is A Bug

This violates API consistency principles in three critical ways:

1. **Semantic Mismatch**: `NotImplementedError` incorrectly suggests that non-UTF-8 support is a missing feature that might be added in the future, when it's actually a permanent design constraint. According to Python's exception hierarchy, `NotImplementedError` should be used for abstract methods or unimplemented functionality, not for invalid parameter validation.

2. **Inconsistent Error Handling**: Users must catch two different exception types to handle the same logical error across symmetric operations:
   ```python
   # Current requirement - unnecessarily complex
   try:
       if reading:
           read_clipboard(encoding=enc)
       else:
           to_clipboard(df, encoding=enc)
   except (NotImplementedError, ValueError) as e:
       # Handle encoding error
       pass
   ```

3. **Undocumented Behavior**: The encoding parameter is completely undocumented in both functions' docstrings (lines 28-73 and 135-156). Users have no way to know that the parameter exists, that only UTF-8 is supported, or what exceptions to expect. This makes the inconsistency even more problematic as users discover it only through trial and error.

## Relevant Context

The issue is located in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/io/clipboards.py`:

- **Line 79**: `read_clipboard()` raises `NotImplementedError("reading from clipboard only supports utf-8 encoding")`
- **Line 161**: `to_clipboard()` raises `ValueError("clipboard only supports utf-8 encoding")`

Both functions perform identical validation (lines 78-79 and 160-161):
```python
if encoding is not None and encoding.lower().replace("-", "") != "utf8":
```

The validation logic is identical, but the exception types differ. The code comments confirm this is intentional validation, not a missing feature (lines 76-77: "only utf-8 is valid for passed value because that's what clipboard supports").

This appears to be an oversight rather than intentional design, as there's no technical reason for the inconsistency. The `ValueError` used by `to_clipboard()` is the correct exception type according to Python conventions for invalid parameter values.

## Proposed Fix

```diff
--- a/pandas/io/clipboards.py
+++ b/pandas/io/clipboards.py
@@ -76,7 +76,7 @@ def read_clipboard(
     # only utf-8 is valid for passed value because that's what clipboard
     # supports
     if encoding is not None and encoding.lower().replace("-", "") != "utf8":
-        raise NotImplementedError("reading from clipboard only supports utf-8 encoding")
+        raise ValueError("reading from clipboard only supports utf-8 encoding")

     check_dtype_backend(dtype_backend)
```