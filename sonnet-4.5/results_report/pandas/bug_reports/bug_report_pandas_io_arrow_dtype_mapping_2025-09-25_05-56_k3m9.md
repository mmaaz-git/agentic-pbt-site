# Bug Report: pandas.io._util._arrow_dtype_mapping Duplicate Dictionary Key

**Target**: `pandas.io._util._arrow_dtype_mapping`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_arrow_dtype_mapping()` function in pandas/io/_util.py contains a duplicate dictionary key (`pa.string()` appears on both lines 41 and 44), causing one of the mappings to be silently ignored.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas.io._util as util

def test_arrow_dtype_mapping_no_duplicate_keys():
    pa = util.import_optional_dependency("pyarrow")
    mapping = util._arrow_dtype_mapping()

    expected_unique_keys = 14
    actual_keys = len(mapping)

    assert actual_keys == expected_unique_keys, (
        f"Expected {expected_unique_keys} keys but got {actual_keys}. "
        "Duplicate keys detected in dictionary literal."
    )
```

**Failing input**: N/A (static bug in source code)

## Reproducing the Bug

```python
import pandas.io._util as util

pa = util.import_optional_dependency("pyarrow")
mapping = util._arrow_dtype_mapping()

print(f"Number of keys: {len(mapping)}")
print(f"Expected: 14 unique keys")
print(f"Actual: 13 keys (pa.string() appears twice, second overwrites first)")
```

## Why This Is A Bug

The function defines a dictionary literal with 14 entries (lines 32-45), but because `pa.string()` is used as a key on both line 41 and line 44, Python's dictionary semantics mean the second value overwrites the first. This results in only 13 keys in the returned dictionary. While both entries map to the same value (`pd.StringDtype()`), having duplicate keys indicates either:

1. A copy-paste error where one should be a different type
2. Dead code that serves no purpose

Either way, this violates the principle of least surprise and should be corrected.

## Fix

```diff
--- a/pandas/io/_util.py
+++ b/pandas/io/_util.py
@@ -38,10 +38,9 @@ def _arrow_dtype_mapping() -> dict:
         pa.uint64(): pd.UInt64Dtype(),
         pa.bool_(): pd.BooleanDtype(),
         pa.string(): pd.StringDtype(),
         pa.float32(): pd.Float32Dtype(),
         pa.float64(): pd.Float64Dtype(),
-        pa.string(): pd.StringDtype(),
         pa.large_string(): pd.StringDtype(),
     }
```