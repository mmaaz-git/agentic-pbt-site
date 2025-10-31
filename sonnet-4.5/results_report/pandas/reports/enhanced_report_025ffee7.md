# Bug Report: pandas.io._util._arrow_dtype_mapping() Duplicate Dictionary Key

**Target**: `pandas.io._util._arrow_dtype_mapping()`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_arrow_dtype_mapping()` function contains a duplicate dictionary key `pa.string()` at lines 41 and 44, causing one mapping to be silently overwritten and resulting in dead code.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import ast


def test_no_duplicate_keys_in_dict_literal():
    pandas_file = "/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/io/_util.py"

    with open(pandas_file) as f:
        source = f.read()

    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_arrow_dtype_mapping":
            for child in ast.walk(node):
                if isinstance(child, ast.Dict):
                    keys_repr = [ast.unparse(k) for k in child.keys]
                    assert len(keys_repr) == len(set(keys_repr)), \
                        f"Duplicate keys found: {keys_repr}"


if __name__ == "__main__":
    test_no_duplicate_keys_in_dict_literal()
```

<details>

<summary>
**Failing input**: The function source code itself contains the duplicate key
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/26/hypo.py", line 23, in <module>
    test_no_duplicate_keys_in_dict_literal()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/26/hypo.py", line 18, in test_no_duplicate_keys_in_dict_literal
    assert len(keys_repr) == len(set(keys_repr)), \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Duplicate keys found: ['pa.int8()', 'pa.int16()', 'pa.int32()', 'pa.int64()', 'pa.uint8()', 'pa.uint16()', 'pa.uint32()', 'pa.uint64()', 'pa.bool_()', 'pa.string()', 'pa.float32()', 'pa.float64()', 'pa.string()', 'pa.large_string()']
```
</details>

## Reproducing the Bug

```python
import ast
import sys

# Read the actual pandas source file to analyze it
pandas_file = "/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/io/_util.py"

with open(pandas_file, 'r') as f:
    source = f.read()

# Parse the source code
tree = ast.parse(source)

# Find the _arrow_dtype_mapping function
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "_arrow_dtype_mapping":
        print(f"Found function: {node.name}")
        # Look for dictionary literals in the function
        for child in ast.walk(node):
            if isinstance(child, ast.Dict):
                # Extract the keys
                keys = []
                for k in child.keys:
                    if isinstance(k, ast.Call):
                        # Get the function call representation
                        keys.append(ast.unparse(k))

                print(f"\nTotal keys in dictionary: {len(keys)}")
                print(f"Unique keys: {len(set(keys))}")

                # Check for duplicates
                seen = set()
                duplicates = []
                for key in keys:
                    if key in seen:
                        duplicates.append(key)
                    else:
                        seen.add(key)

                if duplicates:
                    print(f"\nDuplicate keys found: {duplicates}")
                    print("\nAll keys in order:")
                    for i, key in enumerate(keys, 1):
                        print(f"  {i:2}. {key}")
                else:
                    print("No duplicate keys found")
```

<details>

<summary>
Duplicate key found at positions 10 and 13 in the dictionary
</summary>
```
Found function: _arrow_dtype_mapping

Total keys in dictionary: 14
Unique keys: 13

Duplicate keys found: ['pa.string()']

All keys in order:
   1. pa.int8()
   2. pa.int16()
   3. pa.int32()
   4. pa.int64()
   5. pa.uint8()
   6. pa.uint16()
   7. pa.uint32()
   8. pa.uint64()
   9. pa.bool_()
  10. pa.string()
  11. pa.float32()
  12. pa.float64()
  13. pa.string()
  14. pa.large_string()
```
</details>

## Why This Is A Bug

In Python, when a dictionary literal contains duplicate keys, the later value silently overwrites the earlier one. This violates the principle that all code should have a purpose - the first `pa.string(): pd.StringDtype()` entry at line 41 is dead code that never takes effect because it's immediately overwritten by the identical entry at line 44.

While both duplicate entries map to the same value (pd.StringDtype()), this represents:
1. **Dead code** - Line 41 serves no purpose and is never used
2. **Unclear intent** - The duplicate suggests possible missing functionality, especially given that the related `_arrow_string_types_mapper()` function includes support for `pa.string_view()`
3. **Code quality issue** - Duplicate keys in dictionary literals violate Python best practices and reduce maintainability

The Python documentation states that dictionary comprehensions and literals should not have duplicate keys, and static analysis tools commonly flag this as an issue.

## Relevant Context

The `_arrow_dtype_mapping()` function is an internal pandas function that creates a mapping from PyArrow data types to pandas nullable extension dtypes. It's used when converting PyArrow tables to pandas DataFrames with nullable dtypes.

Looking at the related `_arrow_string_types_mapper()` function in the same file (lines 49-59), we can see it handles three PyArrow string types:
- `pa.string()` - UTF8 variable-length string with 32-bit offsets
- `pa.large_string()` - UTF8 variable-length string with 64-bit offsets
- `pa.string_view()` - UTF8 variable-length string view (PyArrow 18.0+)

The duplicate `pa.string()` entry in `_arrow_dtype_mapping()` might have been intended as `pa.string_view()`, though both would map to the same pandas StringDtype.

Source file location: `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/io/_util.py`

## Proposed Fix

Remove the duplicate key at line 41 since it's overwritten by the identical entry at line 44:

```diff
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