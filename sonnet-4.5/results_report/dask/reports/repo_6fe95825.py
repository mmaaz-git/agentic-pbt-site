#!/usr/bin/env python3
"""Minimal reproduction of dask.utils.parse_bytes bug with whitespace-only strings."""

import dask.utils

# Test empty string
print("Testing empty string '':")
try:
    result = dask.utils.parse_bytes('')
    print(f"  Result: {result}")
except ValueError as e:
    print(f"  ValueError: {e}")

# Test carriage return
print("\nTesting carriage return '\\r':")
try:
    result = dask.utils.parse_bytes('\r')
    print(f"  Result: {result}")
except ValueError as e:
    print(f"  ValueError: {e}")

# Test newline
print("\nTesting newline '\\n':")
try:
    result = dask.utils.parse_bytes('\n')
    print(f"  Result: {result}")
except ValueError as e:
    print(f"  ValueError: {e}")

# Test tab
print("\nTesting tab '\\t':")
try:
    result = dask.utils.parse_bytes('\t')
    print(f"  Result: {result}")
except ValueError as e:
    print(f"  ValueError: {e}")

# Test space
print("\nTesting space ' ':")
try:
    result = dask.utils.parse_bytes(' ')
    print(f"  Result: {result}")
except ValueError as e:
    print(f"  ValueError: {e}")

# Test multiple spaces
print("\nTesting multiple spaces '   ':")
try:
    result = dask.utils.parse_bytes('   ')
    print(f"  Result: {result}")
except ValueError as e:
    print(f"  ValueError: {e}")

# Control: Test invalid unit (should raise ValueError)
print("\nControl test with invalid unit '5 foos':")
try:
    result = dask.utils.parse_bytes('5 foos')
    print(f"  Result: {result}")
except ValueError as e:
    print(f"  ValueError: {e}")