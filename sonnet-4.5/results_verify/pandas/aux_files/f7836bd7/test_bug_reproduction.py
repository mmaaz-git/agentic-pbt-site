#!/usr/bin/env python3
"""Test reproduction for pandas.io.excel._util._excel2num bug report"""

import sys
from pandas.io.excel._util import _excel2num

print("Testing _excel2num function with whitespace inputs...")

# Test 1: Single space
print("\n1. Testing with single space ' ':")
try:
    result = _excel2num(' ')
    print(f"   Result: {result}")
    print(f"   Expected: ValueError")
    print(f"   Actual: Returns {result} (no exception raised)")
except ValueError as e:
    print(f"   Raised ValueError as expected: {e}")

# Test 2: Empty string
print("\n2. Testing with empty string '':")
try:
    result = _excel2num('')
    print(f"   Result: {result}")
    print(f"   Expected: ValueError")
    print(f"   Actual: Returns {result} (no exception raised)")
except ValueError as e:
    print(f"   Raised ValueError as expected: {e}")

# Test 3: Tab character
print("\n3. Testing with tab character '\\t':")
try:
    result = _excel2num('\t')
    print(f"   Result: {result}")
    print(f"   Expected: ValueError")
    print(f"   Actual: Returns {result} (no exception raised)")
except ValueError as e:
    print(f"   Raised ValueError as expected: {e}")

# Test 4: Newline character
print("\n4. Testing with newline character '\\n':")
try:
    result = _excel2num('\n')
    print(f"   Result: {result}")
    print(f"   Expected: ValueError")
    print(f"   Actual: Returns {result} (no exception raised)")
except ValueError as e:
    print(f"   Raised ValueError as expected: {e}")

# Test 5: Multiple spaces
print("\n5. Testing with multiple spaces '   ':")
try:
    result = _excel2num('   ')
    print(f"   Result: {result}")
    print(f"   Expected: ValueError")
    print(f"   Actual: Returns {result} (no exception raised)")
except ValueError as e:
    print(f"   Raised ValueError as expected: {e}")

# Test 6: For comparison - test that other invalid inputs DO raise ValueError
print("\n6. Testing with invalid input 'A B' (contains space):")
try:
    result = _excel2num('A B')
    print(f"   Result: {result}")
    print(f"   Expected: ValueError")
    print(f"   Actual: Returns {result} (no exception raised)")
except ValueError as e:
    print(f"   Raised ValueError as expected: {e}")

print("\n7. Testing with invalid input '123' (numbers):")
try:
    result = _excel2num('123')
    print(f"   Result: {result}")
    print(f"   Expected: ValueError")
    print(f"   Actual: Returns {result} (no exception raised)")
except ValueError as e:
    print(f"   Raised ValueError as expected: {e}")

# Test 8: Valid inputs for comparison
print("\n8. Testing with valid input 'A' (should work):")
try:
    result = _excel2num('A')
    print(f"   Result: {result} (expected: 0)")
except ValueError as e:
    print(f"   Unexpected error: {e}")

print("\n9. Testing with valid input 'AB' (should work):")
try:
    result = _excel2num('AB')
    print(f"   Result: {result} (expected: 27)")
except ValueError as e:
    print(f"   Unexpected error: {e}")