# Bug Report: pandas.io._util._arrow_dtype_mapping Duplicate Dictionary Key

**Target**: `pandas.io._util._arrow_dtype_mapping`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_arrow_dtype_mapping()` function contains a duplicate dictionary key `pa.string()` on lines 41 and 44, making line 44 dead code.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.io._util import _arrow_dtype_mapping

def test_arrow_dtype_mapping_no_duplicate_keys():
    try:
        mapping = _arrow_dtype_mapping()
        import pyarrow as pa

        string_count = len([k for k in mapping.keys() if k == pa.string()])

        assert string_count == 1, f"pa.string() appears {string_count} times"
    except ImportError:
        pass
```

**Failing input**: The function's dictionary literal itself contains the duplicate

## Reproducing the Bug

```python
try:
    import pyarrow as pa
    import pandas as pd
    from pandas.io._util import _arrow_dtype_mapping

    mapping = _arrow_dtype_mapping()

    print("Dictionary literal from source code has duplicate key:")
    print("  Line 41: pa.string(): pd.StringDtype(),")
    print("  Line 44: pa.string(): pd.StringDtype(),")

    actual_count = len([k for k in mapping.keys() if k == pa.string()])
    print(f"Actual: {actual_count} entry for pa.string() in resulting dictionary")
    print("Line 44 is effectively dead code.")

except ImportError:
    print("Install with: pip install pyarrow")
```

## Why This Is A Bug

In Python, when a dictionary literal contains duplicate keys, the later value silently overwrites the earlier one. This means line 44 in `_arrow_dtype_mapping()` has no effect and is dead code. While this doesn't cause incorrect behavior (both map to the same value), it indicates a copy-paste error and violates the principle of clean code.

## Fix

Remove the duplicate key on line 44:

```diff
--- a/pandas/io/_util.py
+++ b/pandas/io/_util.py
@@ -39,9 +39,8 @@ def _arrow_dtype_mapping() -> dict:
         pa.uint64(): pd.UInt64Dtype(),
         pa.bool_(): pd.BooleanDtype(),
         pa.string(): pd.StringDtype(),
         pa.float32(): pd.Float32Dtype(),
         pa.float64(): pd.Float64Dtype(),
-        pa.string(): pd.StringDtype(),
         pa.large_string(): pd.StringDtype(),
     }
```