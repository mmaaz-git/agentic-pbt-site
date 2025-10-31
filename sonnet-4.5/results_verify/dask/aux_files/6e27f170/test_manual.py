from dask.utils import format_bytes

n = 1_125_894_277_343_089_729
result = format_bytes(n)

print(f"Input: {n}")
print(f"Input < 2**60: {n < 2**60}")
print(f"Result: {result!r}")
print(f"Length: {len(result)}")

# Additional tests mentioned in the bug report
test_cases = [
    (999 * 2**50, "999 * 2**50"),
    (1000 * 2**50, "1000 * 2**50"),
    (1024 * 2**50 - 1, "1024 * 2**50 - 1"),
]

print("\nAdditional test cases:")
for value, description in test_cases:
    result = format_bytes(value)
    print(f"  {description:20} -> {result!r:15} ({len(result)} chars)")

# Verify the assertion failures
print("\nAssertion checks:")
n = 1_125_894_277_343_089_729
result = format_bytes(n)
try:
    assert n < 2**60
    print(f"✓ n < 2**60: {n < 2**60}")
except AssertionError:
    print(f"✗ n < 2**60 failed")

try:
    assert len(result) <= 10
    print(f"✓ len(result) <= 10: True")
except AssertionError:
    print(f"✗ len(result) <= 10 failed: len={len(result)} for result='{result}'")