#!/usr/bin/env python3
"""
Minimal reproduction of numpy.base_repr padding bug with zero.
"""
import numpy as np

print("Testing numpy.base_repr padding behavior with number=0")
print("=" * 60)

# Test case 1: The specific failing case from the bug report
print("\nTest 1: number=0, padding=1")
result = np.base_repr(0, padding=1)
print(f"  np.base_repr(0, padding=1) = '{result}'")
print(f"  Expected: '00' (the digit '0' with 1 zero padded on the left)")
print(f"  Got:      '{result}'")
print(f"  Length: {len(result)} (expected: 2)")

# Test case 2: Compare with non-zero number
print("\nTest 2: Comparison with number=1, padding=1")
result_1 = np.base_repr(1, padding=1)
print(f"  np.base_repr(1, padding=1) = '{result_1}'")
print(f"  This correctly adds 1 zero to the left of '1'")

# Test case 3: Multiple padding values with zero
print("\nTest 3: Various padding values with number=0")
for padding in range(0, 5):
    result = np.base_repr(0, padding=padding)
    expected_length = 1 + padding  # '0' plus padding zeros
    print(f"  padding={padding}: '{result}' (length={len(result)}, expected={expected_length})")

# Test case 4: Show the inconsistency clearly
print("\nTest 4: Demonstrating the inconsistency")
print("  For any number N and padding P, we expect:")
print("  len(base_repr(N, padding=P)) = len(base_repr(N, padding=0)) + P")
print()

for num in [0, 1, 5, 10]:
    base_repr_no_pad = np.base_repr(num, padding=0)
    base_repr_with_pad = np.base_repr(num, padding=1)
    expected_len = len(base_repr_no_pad) + 1
    actual_len = len(base_repr_with_pad)
    status = "✓" if expected_len == actual_len else "✗"
    print(f"  num={num:2d}: without padding='{base_repr_no_pad}' (len={len(base_repr_no_pad)})")
    print(f"         with padding=1 ='{base_repr_with_pad}' (len={actual_len}, expected={expected_len}) {status}")

# The assertion that fails
print("\n" + "=" * 60)
print("Assertion test:")
try:
    result = np.base_repr(0, padding=1)
    assert result == '00', f"Expected '00', got '{result}'"
    print("PASS: np.base_repr(0, padding=1) == '00'")
except AssertionError as e:
    print(f"FAIL: {e}")