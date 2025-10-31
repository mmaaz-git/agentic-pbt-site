# Bug Report: pandas.io._util Duplicate Dictionary Key

**Target**: `pandas.io._util._arrow_dtype_mapping()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_arrow_dtype_mapping()` function contains a duplicate dictionary key `pa.string()` at lines 41 and 44, causing one mapping to be silently lost.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import ast


def test_no_duplicate_keys_in_dict_literal():
    with open("/path/to/pandas/io/_util.py") as f:
        source = f.read()

    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_arrow_dtype_mapping":
            for child in ast.walk(node):
                if isinstance(child, ast.Dict):
                    keys_repr = [ast.unparse(k) for k in child.keys]
                    assert len(keys_repr) == len(set(keys_repr)), \
                        f"Duplicate keys found: {keys_repr}"
```

**Failing input**: The function source code itself contains the bug.

## Reproducing the Bug

```python
import ast

source = """
def _arrow_dtype_mapping() -> dict:
    pa = import_optional_dependency("pyarrow")
    return {
        pa.int8(): pd.Int8Dtype(),
        pa.int16(): pd.Int16Dtype(),
        pa.int32(): pd.Int32Dtype(),
        pa.int64(): pd.Int64Dtype(),
        pa.uint8(): pd.UInt8Dtype(),
        pa.uint16(): pd.UInt16Dtype(),
        pa.uint32(): pd.UInt32Dtype(),
        pa.uint64(): pd.UInt64Dtype(),
        pa.bool_(): pd.BooleanDtype(),
        pa.string(): pd.StringDtype(),
        pa.float32(): pd.Float32Dtype(),
        pa.float64(): pd.Float64Dtype(),
        pa.string(): pd.StringDtype(),
        pa.large_string(): pd.StringDtype(),
    }
"""

tree = ast.parse(source)
for node in ast.walk(tree):
    if isinstance(node, ast.Dict):
        keys = [ast.unparse(k) for k in node.keys]
        print(f"Total keys: {len(keys)}")
        print(f"Unique keys: {len(set(keys))}")
        if len(keys) != len(set(keys)):
            print("Duplicate key found: pa.string()")
```

## Why This Is A Bug

In Python, when a dictionary literal contains duplicate keys, the later value overwrites the earlier one. Lines 41 and 44 both define `pa.string(): pd.StringDtype()`, meaning the entry at line 41 has no effect and is dead code. This indicates either:

1. Line 41 was meant to be removed (since line 44 has the same mapping)
2. One of the lines should map a different PyArrow type (the pattern suggests `pa.string_view()` might have been intended, as it appears in the related `_arrow_string_types_mapper()` function)

The duplicate reduces the effectiveness of the type mapping and represents unclear intent in the code.

## Fix

Remove the duplicate key at line 41:

```diff
diff --git a/pandas/io/_util.py b/pandas/io/_util.py
index 1234567..abcdefg 100644
--- a/pandas/io/_util.py
+++ b/pandas/io/_util.py
@@ -38,7 +38,6 @@ def _arrow_dtype_mapping() -> dict:
         pa.uint32(): pd.UInt32Dtype(),
         pa.uint64(): pd.UInt64Dtype(),
         pa.bool_(): pd.BooleanDtype(),
-        pa.string(): pd.StringDtype(),
         pa.float32(): pd.Float32Dtype(),
         pa.float64(): pd.Float64Dtype(),
         pa.string(): pd.StringDtype(),
```