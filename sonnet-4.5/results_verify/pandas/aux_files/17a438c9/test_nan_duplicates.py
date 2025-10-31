#!/usr/bin/env python3

import math
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from pandas.io.parsers.readers import _validate_names

# Test 1: Basic reproduction of the bug
print("Test 1: Basic reproduction with duplicate NaN values")
print("=" * 60)

names = [float('nan'), float('nan')]
print(f"Input: {names}")
print(f"len(names): {len(names)}")
print(f"len(set(names)): {len(set(names))}")
print(f"set(names): {set(names)}")

try:
    _validate_names(names)
    print("Result: _validate_names accepted duplicate NaN values (no exception raised)")
except ValueError as e:
    print(f"Result: _validate_names rejected duplicate NaN values with error: {e}")

print()

# Test 2: Test with more than 2 NaN values
print("Test 2: Three NaN values")
print("=" * 60)

names_3nan = [float('nan'), float('nan'), float('nan')]
print(f"Input: {names_3nan}")
print(f"len(names): {len(names_3nan)}")
print(f"len(set(names)): {len(set(names_3nan))}")

try:
    _validate_names(names_3nan)
    print("Result: _validate_names accepted three NaN values (no exception raised)")
except ValueError as e:
    print(f"Result: _validate_names rejected three NaN values with error: {e}")

print()

# Test 3: Test with mixed NaN and regular values
print("Test 3: Mixed NaN and regular values")
print("=" * 60)

names_mixed = [float('nan'), 'col1', float('nan'), 'col2']
print(f"Input: {names_mixed}")
print(f"len(names): {len(names_mixed)}")
print(f"len(set(names)): {len(set(names_mixed))}")

try:
    _validate_names(names_mixed)
    print("Result: _validate_names accepted mixed NaN and regular values (no exception raised)")
except ValueError as e:
    print(f"Result: _validate_names rejected mixed values with error: {e}")

print()

# Test 4: Test with regular duplicate values (should raise error)
print("Test 4: Regular duplicate values (control test)")
print("=" * 60)

names_dup = ['col1', 'col1']
print(f"Input: {names_dup}")

try:
    _validate_names(names_dup)
    print("Result: _validate_names accepted duplicate regular values (no exception raised)")
except ValueError as e:
    print(f"Result: _validate_names correctly rejected duplicate regular values with error: {e}")

print()

# Test 5: Verify NaN inequality
print("Test 5: Verify NaN inequality property")
print("=" * 60)
nan1 = float('nan')
nan2 = float('nan')
print(f"nan1 == nan2: {nan1 == nan2}")
print(f"nan1 is nan2: {nan1 is nan2}")
print(f"math.isnan(nan1): {math.isnan(nan1)}")
print(f"math.isnan(nan2): {math.isnan(nan2)}")