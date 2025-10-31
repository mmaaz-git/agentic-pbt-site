#!/usr/bin/env python3
"""Test the reported bug in xarray.core.formatting.pretty_print"""

import xarray.core.formatting as fmt

print("Testing the bug report for xarray.core.formatting.pretty_print")
print("=" * 60)

# Test case 1: numchars=1
print("\nTest 1: numchars=1")
print("-" * 30)
obj = "test"
numchars = 1
result = fmt.pretty_print(obj, numchars)

print(f"Input: obj='{obj}', numchars={numchars}")
print(f"Output: '{result}'")
print(f"Output length: {len(result)}")
print(f"Expected length: {numchars}")
print(f"PASS: {len(result) == numchars}")

# Test case 2: numchars=2
print("\nTest 2: numchars=2")
print("-" * 30)
numchars = 2
result = fmt.pretty_print(obj, numchars)

print(f"Input: obj='{obj}', numchars={numchars}")
print(f"Output: '{result}'")
print(f"Output length: {len(result)}")
print(f"Expected length: {numchars}")
print(f"PASS: {len(result) == numchars}")

# Test case 3: numchars=3
print("\nTest 3: numchars=3")
print("-" * 30)
numchars = 3
result = fmt.pretty_print(obj, numchars)

print(f"Input: obj='{obj}', numchars={numchars}")
print(f"Output: '{result}'")
print(f"Output length: {len(result)}")
print(f"Expected length: {numchars}")
print(f"PASS: {len(result) == numchars}")

# Test case 4: numchars=4 (equal to string length)
print("\nTest 4: numchars=4 (equal to string length)")
print("-" * 30)
numchars = 4
result = fmt.pretty_print(obj, numchars)

print(f"Input: obj='{obj}', numchars={numchars}")
print(f"Output: '{result}'")
print(f"Output length: {len(result)}")
print(f"Expected length: {numchars}")
print(f"PASS: {len(result) == numchars}")

# Test case 5: numchars=5 (greater than string length)
print("\nTest 5: numchars=5 (greater than string length)")
print("-" * 30)
numchars = 5
result = fmt.pretty_print(obj, numchars)

print(f"Input: obj='{obj}', numchars={numchars}")
print(f"Output: '{result}'")
print(f"Output length: {len(result)}")
print(f"Expected length: {numchars}")
print(f"PASS: {len(result) == numchars}")

# Test with longer string
print("\nTest 6: Long string with numchars=10")
print("-" * 30)
obj = "This is a very long string that should be truncated"
numchars = 10
result = fmt.pretty_print(obj, numchars)

print(f"Input: obj='{obj}', numchars={numchars}")
print(f"Output: '{result}'")
print(f"Output length: {len(result)}")
print(f"Expected length: {numchars}")
print(f"PASS: {len(result) == numchars}")

# Also test maybe_truncate directly
print("\n\nTesting maybe_truncate directly:")
print("=" * 60)

for maxlen in [1, 2, 3, 4, 5]:
    print(f"\nmaxlen={maxlen}")
    result = fmt.maybe_truncate("test", maxlen)
    print(f"  Result: '{result}'")
    print(f"  Length: {len(result)} (expected <= {maxlen})")
    print(f"  Valid: {len(result) <= maxlen}")