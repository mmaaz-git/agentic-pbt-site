#!/usr/bin/env python3
"""Test additional edge cases for format_bytes"""

from dask.utils import format_bytes

# Test values around unit boundaries
test_cases = [
    # Around PiB boundary (2**50)
    (2**50 * 0.9 - 1, "Just below PiB threshold"),
    (2**50 * 0.9, "At PiB threshold"),
    (2**50 * 0.999, "Close to 1000 PiB"),
    (2**50 * 0.9999, "Very close to 1000 PiB"),
    (2**50 * 0.99999, "Even closer to 1000 PiB"),
    (2**50 - 1, "Just below 1 PiB"),

    # Around TiB boundary (2**40)
    (2**40 * 0.9999, "Close to 1000 TiB"),
    (2**40 * 0.99999, "Very close to 1000 TiB"),

    # Test the upper bound
    (2**60 - 1, "Maximum value < 2**60"),
]

print("Testing edge cases for format_bytes:")
print("-" * 60)

violations = []
for value, description in test_cases:
    result = format_bytes(value)
    length = len(result)
    valid = length <= 10

    if not valid:
        violations.append((value, result, length))

    status = "✓" if valid else "✗"
    print(f"{status} {description:30} => {result:12} (len={length})")

if violations:
    print("\n" + "=" * 60)
    print(f"Found {len(violations)} violation(s) of the 10-character guarantee:")
    for value, result, length in violations:
        print(f"  format_bytes({value}) = {result!r} (length {length})")
else:
    print("\nAll test cases passed!")

# Test that values >= 2**60 work (even though no guarantee)
print("\n" + "-" * 60)
print("Testing values >= 2**60 (no guarantee):")
for exp in [60, 61, 62]:
    value = 2**exp
    result = format_bytes(value)
    print(f"  2**{exp}: {result} (len={len(result)})")