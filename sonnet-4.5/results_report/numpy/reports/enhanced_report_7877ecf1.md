# Bug Report: numpy.char.mod Fails to Handle Tuple Arguments for Multiple Format Specifiers

**Target**: `numpy.char.mod`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.char.mod` incorrectly wraps tuple arguments in an extra tuple layer, causing format strings with multiple placeholders to fail while Python's built-in `%` operator handles them correctly.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test for numpy.char.mod tuple handling bug.
This test demonstrates that numpy.char.mod fails to handle tuple arguments
for format strings with multiple placeholders.
"""

import numpy as np
import numpy.char as char
from hypothesis import given, strategies as st

@given(st.integers(min_value=0, max_value=100))
def test_mod_with_multiple_formats(value):
    """Test that numpy.char.mod handles tuples for multiple format specifiers."""
    # Create a format string with two format specifiers
    format_string = 'value: %d, hex: %x'
    arr = np.array([format_string], dtype=str)

    # This should work the same as Python's % operator
    expected = format_string % (value, value)

    # But numpy.char.mod fails with a tuple
    result = char.mod(arr, (value, value))

    assert result[0] == expected, f"Expected '{expected}' but got '{result[0]}'"

if __name__ == "__main__":
    # Run the test with a simple value to demonstrate the failure
    print("Testing numpy.char.mod with tuple for multiple format specifiers...")
    print("\nRunning property-based test with value=0:")
    format_string = 'value: %d, hex: %x'
    print(f"Python's % operator: '{format_string}' % (0, 0) = '{format_string % (0, 0)}'")
    try:
        arr = np.array([format_string], dtype=str)
        result = char.mod(arr, (0, 0))
        print(f"numpy.char.mod: {result}")
    except Exception as e:
        print(f"numpy.char.mod raised: {type(e).__name__}: {e}")

    # Also run the hypothesis test
    print("\nRunning hypothesis test:")
    try:
        test_mod_with_multiple_formats()
        print("Test passed!")
    except Exception as e:
        print(f"Test failed: {type(e).__name__}: {e}")
```

<details>

<summary>
**Failing input**: `value=0`
</summary>
```
Testing numpy.char.mod with tuple for multiple format specifiers...

Running property-based test with value=0:
Python's % operator: 'value: %d, hex: %x' % (0, 0) = 'value: 0, hex: 0'
numpy.char.mod raised: TypeError: not enough arguments for format string

Running hypothesis test:
Test failed: TypeError: not enough arguments for format string
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction of numpy.char.mod bug with tuple arguments.
This demonstrates that numpy.char.mod fails to handle tuple arguments
for format strings with multiple placeholders, while Python's built-in
% operator handles them correctly.
"""

import numpy as np
import numpy.char as char

# Test 1: Python's % operator works correctly with tuple for multiple formats
print("=== Python's built-in % operator ===")
python_result = 'x=%d, y=%d' % (5, 10)
print(f"'x=%d, y=%d' % (5, 10) = '{python_result}'")
print()

# Test 2: numpy.char.mod fails with tuple for multiple formats
print("=== numpy.char.mod with tuple for multiple formats ===")
try:
    arr = np.array(['x=%d, y=%d'], dtype=str)
    result = char.mod(arr, (5, 10))
    print(f"char.mod(['x=%d, y=%d'], (5, 10)) = {result}")
except Exception as e:
    print(f"char.mod(['x=%d, y=%d'], (5, 10)) raised: {type(e).__name__}: {e}")
print()

# Test 3: Show that single format works
print("=== numpy.char.mod with single format (works) ===")
try:
    arr_single = np.array(['x=%d'], dtype=str)
    result_single = char.mod(arr_single, 5)
    print(f"char.mod(['x=%d'], 5) = {result_single}")
except Exception as e:
    print(f"char.mod(['x=%d'], 5) raised: {type(e).__name__}: {e}")
print()

# Test 4: Show that dict-based formatting works
print("=== numpy.char.mod with dict formatting (works) ===")
try:
    arr_dict = np.array(['%(x)d, %(y)d'], dtype=str)
    result_dict = char.mod(arr_dict, {'x': 5, 'y': 10})
    print(f"char.mod(['%(x)d, %(y)d'], {{'x': 5, 'y': 10}}) = {result_dict}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
```

<details>

<summary>
TypeError: not enough arguments for format string
</summary>
```
=== Python's built-in % operator ===
'x=%d, y=%d' % (5, 10) = 'x=5, y=10'

=== numpy.char.mod with tuple for multiple formats ===
char.mod(['x=%d, y=%d'], (5, 10)) raised: TypeError: not enough arguments for format string

=== numpy.char.mod with single format (works) ===
char.mod(['x=%d'], 5) = ['x=5']

=== numpy.char.mod with dict formatting (works) ===
char.mod(['%(x)d, %(y)d'], {'x': 5, 'y': 10}) = ['5, 10']
```
</details>

## Why This Is A Bug

This violates NumPy's documented behavior in multiple ways:

1. **Documentation contract violation**: The `numpy.char.mod` documentation explicitly states it performs "pre-Python 2.6 string formatting (interpolation)" and returns "(a % i)". Python's `%` operator has always supported tuple arguments for multiple format specifiers - this is fundamental behavior that predates Python 2.6.

2. **Inconsistent behavior**: The function correctly handles:
   - Single format specifiers with non-tuple values: `char.mod(['x=%d'], 5)` works
   - Dictionary-based formatting: `char.mod(['%(x)d, %(y)d'], {'x': 5, 'y': 10})` works
   - But fails for the standard tuple case: `char.mod(['x=%d, y=%d'], (5, 10))` fails

3. **Root cause identified**: The implementation at `/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/strings.py:268` wraps the `values` parameter in `(values,)`, which adds an unwanted extra tuple layer. This transforms `'x=%d, y=%d'.__mod__((5, 10))` into `'x=%d, y=%d'.__mod__(((5, 10),))`, causing the TypeError.

4. **Common usage pattern broken**: Tuple-based formatting with multiple specifiers is a widely-used Python idiom that users reasonably expect to work based on NumPy's claim of implementing Python's % operator behavior.

## Relevant Context

- **NumPy documentation**: https://numpy.org/doc/stable/reference/generated/numpy.char.mod.html states the function performs "pre-Python 2.6 string formatting"
- **Python % operator documentation**: https://docs.python.org/3/library/stdtypes.html#printf-style-string-formatting shows tuple usage for multiple format specifiers
- **Source code location**: `/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/strings.py:268`
- **Workaround available**: Dictionary-based formatting works correctly, e.g., `char.mod(['%(x)d, %(y)d'], {'x': 5, 'y': 10})`
- **NumPy version affected**: Current version (as of 2025-09-25)

The bug affects element-wise string operations on arrays, making it impossible to use the common Python pattern of tuple arguments for format strings with multiple placeholders in vectorized NumPy operations.

## Proposed Fix

```diff
--- a/numpy/_core/strings.py
+++ b/numpy/_core/strings.py
@@ -265,7 +265,10 @@ def mod(a, values):

     """
     return _to_bytes_or_str_array(
-        _vec_string(a, np.object_, '__mod__', (values,)), a)
+        _vec_string(a, np.object_, '__mod__',
+                    values if isinstance(values, tuple) else (values,)), a)


 @set_module("numpy.strings")
```