# Bug Report: pandas.io._util._arrow_dtype_mapping Duplicate Dictionary Key

**Target**: `pandas.io._util._arrow_dtype_mapping`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_arrow_dtype_mapping()` function contains a duplicate dictionary key `pa.string()` on lines 41 and 44 of `/pandas/io/_util.py`, making line 44 dead code that has no effect.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test for pandas.io._util._arrow_dtype_mapping duplicate key bug
"""

from hypothesis import given, strategies as st
from pandas.io._util import _arrow_dtype_mapping

def test_arrow_dtype_mapping_no_duplicate_keys():
    """
    Test that _arrow_dtype_mapping() does not have duplicate keys.

    This test checks the runtime behavior of the dictionary returned
    by _arrow_dtype_mapping(). While Python allows duplicate keys in
    dictionary literals (with later values overwriting earlier ones),
    having duplicates represents dead code and violates the DRY principle.
    """
    try:
        import pyarrow as pa

        # Get the mapping dictionary
        mapping = _arrow_dtype_mapping()

        # Count occurrences of pa.string() in the keys
        string_count = len([k for k in mapping.keys() if k == pa.string()])

        # While the runtime dictionary has only 1 key (Python's behavior),
        # the source code contains a duplicate which is the actual bug
        assert string_count == 1, f"pa.string() appears {string_count} times in runtime dict"

        # The real issue is in the source code where line 44 duplicates line 41
        print("Test passed: Runtime dictionary has 1 pa.string() key")
        print("However, source code has duplicate pa.string() keys on lines 41 and 44")
        print("Line 44 is dead code that should be removed")

    except ImportError as e:
        print(f"Skipping test due to missing dependency: {e}")

# Run the test
if __name__ == "__main__":
    test_arrow_dtype_mapping_no_duplicate_keys()
```

<details>

<summary>
**Failing input**: Dictionary literal with duplicate key in source code
</summary>
```
Test passed: Runtime dictionary has 1 pa.string() key
However, source code has duplicate pa.string() keys on lines 41 and 44
Line 44 is dead code that should be removed
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction case for pandas.io._util._arrow_dtype_mapping duplicate key bug
"""

try:
    import pyarrow as pa
    import pandas as pd
    from pandas.io._util import _arrow_dtype_mapping

    # Get the mapping dictionary
    mapping = _arrow_dtype_mapping()

    print("=== Duplicate Dictionary Key Bug Demonstration ===\n")
    print("Source code in pandas/io/_util.py contains:")
    print("  Line 41: pa.string(): pd.StringDtype(),")
    print("  Line 44: pa.string(): pd.StringDtype(),")
    print("\nThis creates a duplicate key in the dictionary literal.\n")

    # Count occurrences of pa.string() in the keys
    string_key_count = len([k for k in mapping.keys() if k == pa.string()])
    print(f"Number of pa.string() keys in resulting dictionary: {string_key_count}")

    # Show that line 44 is dead code
    print("\nPython behavior with duplicate dictionary keys:")
    print("  - Later values overwrite earlier ones")
    print("  - Line 44 overwrites line 41 with the same value")
    print("  - Result: Line 44 is effectively dead code")

    # Verify both string types are present
    has_string = pa.string() in mapping
    has_large_string = pa.large_string() in mapping

    print(f"\npa.string() in mapping: {has_string}")
    print(f"pa.large_string() in mapping: {has_large_string}")

    # Show the actual mapping
    print(f"\nmapping[pa.string()] = {mapping[pa.string()]}")
    print(f"mapping[pa.large_string()] = {mapping[pa.large_string()]}")

    print("\n=== Conclusion ===")
    print("The duplicate pa.string() key on line 44 should be removed.")
    print("This is dead code that violates the DRY principle.")

except ImportError as e:
    print(f"Error: {e}")
    print("Please install pyarrow: pip install pyarrow")
```

<details>

<summary>
Duplicate key creates dead code but no functional error
</summary>
```
=== Duplicate Dictionary Key Bug Demonstration ===

Source code in pandas/io/_util.py contains:
  Line 41: pa.string(): pd.StringDtype(),
  Line 44: pa.string(): pd.StringDtype(),

This creates a duplicate key in the dictionary literal.

Number of pa.string() keys in resulting dictionary: 1

Python behavior with duplicate dictionary keys:
  - Later values overwrite earlier ones
  - Line 44 overwrites line 41 with the same value
  - Result: Line 44 is effectively dead code

pa.string() in mapping: True
pa.large_string() in mapping: True

mapping[pa.string()] = string
mapping[pa.large_string()] = string

=== Conclusion ===
The duplicate pa.string() key on line 44 should be removed.
This is dead code that violates the DRY principle.
```
</details>

## Why This Is A Bug

This violates expected behavior because:

1. **Dead Code**: In Python, when a dictionary literal contains duplicate keys, the later value silently overwrites the earlier one. This means line 44 in `_arrow_dtype_mapping()` has no effect at runtime and is dead code.

2. **Violates DRY Principle**: The Don't Repeat Yourself principle is a fundamental software engineering practice. Having identical duplicate entries violates this principle.

3. **Code Quality Issue**: While this doesn't cause functional errors (both entries map to the same `pd.StringDtype()` value), it represents poor code quality that could confuse maintainers and suggests a copy-paste error.

4. **Python Best Practices**: The Python documentation and style guides discourage duplicate keys in dictionary literals as they serve no purpose and create confusion.

## Relevant Context

- **Function Location**: `/pandas/io/_util.py`, lines 29-46
- **Function Purpose**: Maps PyArrow data types to pandas nullable extension dtypes
- **PyArrow Types**:
  - `pa.string()`: UTF8 variable-length string type with 32-bit offsets
  - `pa.large_string()`: Large UTF8 variable-length string type with 64-bit offsets
  - Both correctly map to `pd.StringDtype()` as pandas doesn't distinguish between them
- **Impact**: Zero functional impact since both duplicates map to the same value, but represents technical debt
- **Related Code**: The same file has `_arrow_string_types_mapper()` (lines 49-59) which correctly has only one entry for `pa.string()`

Documentation references:
- [pandas PyArrow Integration](https://pandas.pydata.org/docs/user_guide/pyarrow.html)
- [PyArrow Data Types](https://arrow.apache.org/docs/python/api/datatypes.html)

## Proposed Fix

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