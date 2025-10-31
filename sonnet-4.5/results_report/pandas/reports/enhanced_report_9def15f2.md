# Bug Report: pandas.io.excel.inspect_excel_format Empty Stream Contract Violation

**Target**: `pandas.io.excel._base.inspect_excel_format`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `inspect_excel_format` function returns `None` when given an empty byte stream, but its docstring explicitly states it should raise `ValueError("stream is empty")` in this case.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.io.excel._base import inspect_excel_format
import pytest


def test_inspect_excel_format_empty_raises():
    """Test that inspect_excel_format raises ValueError on empty input as documented."""
    with pytest.raises(ValueError, match="stream is empty"):
        inspect_excel_format(b'')
```

<details>

<summary>
**Failing input**: `b''`
</summary>
```
Running test: inspect_excel_format with empty bytes...

Test FAILED: No exception raised
Function returned: None
Expected: ValueError("stream is empty")
```
</details>

## Reproducing the Bug

```python
from pandas.io.excel._base import inspect_excel_format

# Test with empty bytes
print("Testing inspect_excel_format with empty bytes b''")
print("According to docstring, this should raise ValueError('stream is empty')")
print()

try:
    result = inspect_excel_format(b'')
    print(f"Result: {result}")
    print(f"Type of result: {type(result)}")
    print("ERROR: No exception was raised! Expected ValueError.")
except ValueError as e:
    print(f"ValueError raised as expected: {e}")
except Exception as e:
    print(f"Unexpected exception: {type(e).__name__}: {e}")
```

<details>

<summary>
Function returns None instead of raising ValueError
</summary>
```
Testing inspect_excel_format with empty bytes b''
According to docstring, this should raise ValueError('stream is empty')

Result: None
Type of result: <class 'NoneType'>
ERROR: No exception was raised! Expected ValueError.
```
</details>

## Why This Is A Bug

The function's docstring at lines 1392-1395 explicitly documents:

```
Raises
------
ValueError
    If resulting stream is empty.
```

However, the implementation at line 1408 checks `if buf is None:` which is incorrect. When `stream.read(PEEK_SIZE)` is called on an empty stream, it returns empty bytes `b''`, not `None`. Since `b'' is None` evaluates to `False`, the ValueError is never raised and the function continues to line 1417 where it returns `None` because empty bytes don't match any Excel signatures.

This violates the documented contract in three ways:
1. The function should raise an exception for empty input but returns a value instead
2. Callers expecting exception-based error handling will not catch the empty stream case
3. The return value `None` is ambiguous - it could mean either "empty stream" or "unknown format"

## Relevant Context

- The function is in a private module (`_base` with underscore prefix) but has comprehensive documentation forming a contract
- The bug occurs because `BytesIO(b'').read(n)` returns `b''` (empty bytes), not `None`
- Python's truthiness: `bool(b'')` is `False`, but `b'' is None` is also `False`
- The current behavior silently treats empty files as "unknown format" rather than invalid input
- Documentation URL: The function is not in public pandas docs (404 on official documentation site)

## Proposed Fix

```diff
--- a/pandas/io/excel/_base.py
+++ b/pandas/io/excel/_base.py
@@ -1405,7 +1405,7 @@ def inspect_excel_format(
         stream = handle.handle
         stream.seek(0)
         buf = stream.read(PEEK_SIZE)
-        if buf is None:
+        if not buf:
             raise ValueError("stream is empty")
         assert isinstance(buf, bytes)
         peek = buf
```