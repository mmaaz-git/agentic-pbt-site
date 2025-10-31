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