# Bug Report: pandas.core.strings.accessor.cat_core Null Byte Separator Silently Dropped

**Target**: `pandas.core.strings.accessor.cat_core`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `cat_core` function and the public `Series.str.cat()` method silently drop null byte (`\x00`) characters when used as separators, resulting in data corruption without any error or warning.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings

import pandas.core.strings.accessor as accessor


@given(
    array_length=st.integers(min_value=1, max_value=10),
    num_arrays=st.integers(min_value=2, max_value=4),
)
@settings(max_examples=500)
def test_cat_core_preserves_separator(array_length, num_arrays):
    sep = '\x00'

    arrays = [
        np.array([f's{i}_{j}' for j in range(array_length)], dtype=object)
        for i in range(num_arrays)
    ]

    result = accessor.cat_core(arrays, sep)

    for i in range(array_length):
        expected = sep.join([arrays[j][i] for j in range(num_arrays)])
        assert result[i] == expected, \
            f"At index {i}: expected {repr(expected)}, got {repr(result[i])}"


if __name__ == "__main__":
    test_cat_core_preserves_separator()
```

<details>

<summary>
**Failing input**: `array_length=1, num_arrays=2`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 29, in <module>
    test_cat_core_preserves_separator()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 8, in test_cat_core_preserves_separator
    array_length=st.integers(min_value=1, max_value=10),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 24, in test_cat_core_preserves_separator
    assert result[i] == expected, \
           ^^^^^^^^^^^^^^^^^^^^^
AssertionError: At index 0: expected 's0_0\x00s1_0', got 's0_0s1_0'
Falsifying example: test_cat_core_preserves_separator(
    array_length=1,  # or any other generated value
    num_arrays=2,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import pandas.core.strings.accessor as accessor

# Create simple test arrays
arr1 = np.array(['hello'], dtype=object)
arr2 = np.array(['world'], dtype=object)

# Use null byte as separator
result = accessor.cat_core([arr1, arr2], '\x00')

# Display the results
print(f"Result: {repr(result[0])}")
print(f"Expected: {repr('hello\x00world')}")
print(f"Null byte present in result: {'\\x00' in result[0]}")
print(f"Length of result: {len(result[0])}")
print(f"Length of expected: {len('hello\x00world')}")

# Test assertion
try:
    assert result[0] == 'hello\x00world', "Null byte was silently dropped!"
except AssertionError as e:
    print(f"\nAssertion failed: {e}")
    print(f"The null byte separator was silently dropped from the concatenation.")
```

<details>

<summary>
Output shows null byte was silently dropped
</summary>
```
Result: 'helloworld'
Expected: 'hello\x00world'
Null byte present in result: False
Length of result: 10
Length of expected: 11

Assertion failed: Null byte was silently dropped!
The null byte separator was silently dropped from the concatenation.
```
</details>

## Why This Is A Bug

This violates expected behavior in several critical ways:

1. **Silent Data Corruption**: The function accepts null bytes without error but produces incorrect output. Data is lost without any warning, making the bug difficult to detect in production systems.

2. **Inconsistent Behavior**: All other special characters work correctly - tabs (`\t`), newlines (`\n`), SOH (`\x01`), and even empty strings. Only null bytes (`\x00`) fail, creating an unexpected special case.

3. **Documentation Violation**: The function documentation states the `sep` parameter accepts a "string" with no restrictions. In Python, strings fully support null bytes as legitimate characters. There is no documented limitation on separator characters.

4. **Affects Public API**: This bug propagates to the public `pandas.Series.str.cat()` method, affecting end users who may rely on null-delimited data formats.

5. **Real-World Impact**: Null bytes are used in various data formats including:
   - Binary protocols with null-delimited fields
   - Database export formats
   - System-level data interchange
   - Legacy data processing pipelines

## Relevant Context

The root cause is an interaction between pandas and NumPy's `np.sum()` function. When `np.sum()` operates on an object array containing mixed types (numpy arrays and scalar strings), it fails to properly handle null bytes in the scalar strings.

Testing shows:
- The bug only affects null bytes (`\x00`) - all other byte values including `\x01` work correctly
- The issue occurs in both `cat_core` and the public `Series.str.cat()` method
- Multiple null bytes are all dropped (e.g., `\x00\x00` becomes empty string)
- Interestingly, null bytes embedded within other characters work (e.g., `'X\x00Y'` as separator works correctly)

Source code location: `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/strings/accessor.py`

## Proposed Fix

Convert the separator string to a numpy array before concatenation to ensure consistent type handling:

```diff
--- a/pandas/core/strings/accessor.py
+++ b/pandas/core/strings/accessor.py
@@ -2650,7 +2650,11 @@ def cat_core(list_of_columns: list, sep: str):
     if sep == "":
         # no need to interleave sep if it is empty
         arr_of_cols = np.asarray(list_of_columns, dtype=object)
         return np.sum(arr_of_cols, axis=0)
-    list_with_sep = [sep] * (2 * len(list_of_columns) - 1)
+
+    # Convert separator string to array to avoid numpy's null byte handling issues
+    sep_array = np.array([sep] * len(list_of_columns[0]), dtype=object)
+
+    list_with_sep = [sep_array] * (2 * len(list_of_columns) - 1)
     list_with_sep[::2] = list_of_columns
     arr_with_sep = np.asarray(list_with_sep, dtype=object)
     return np.sum(arr_with_sep, axis=0)
```