#!/usr/bin/env python3
from dask.utils import format_bytes

# Test the specific failing case
n = 1_125_899_906_842_624_000
result = format_bytes(n)
print(f"Input: {n}")
print(f"Result: '{result}'")
print(f"Length: {len(result)} characters")
print()

# Test boundary cases
test_cases = [
    (999 * 2**50, "999 * 2**50"),
    (1000 * 2**50, "1000 * 2**50"),
    (1023 * 2**50, "1023 * 2**50"),
]

print("Boundary cases:")
for value, description in test_cases:
    result = format_bytes(value)
    print(f"  {description} = '{result}' ({len(result)} chars)")

# Verify the documented guarantee
print()
print("Checking documented guarantee: 'For all values < 2**60, the output is always <= 10 characters'")
print(f"Is {n} < 2**60? {n < 2**60}")
print(f"Is output <= 10 characters? {len(format_bytes(n)) <= 10}")

# Trigger the assertion error
assert len(format_bytes(n)) <= 10, f"Expected <= 10 characters, got {len(format_bytes(n))}"