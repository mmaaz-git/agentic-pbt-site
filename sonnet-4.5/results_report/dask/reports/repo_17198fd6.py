from dask.utils import format_bytes

# Test values around 1000 PiB boundary
pib = 2**50

test_cases = [
    999 * pib,     # Should be 10 chars
    1000 * pib,    # Bug: produces 11 chars
    1023 * pib,    # Bug: produces 11 chars
]

print("Testing format_bytes length guarantee:")
print("Docstring claims: 'For all values < 2**60, the output is always <= 10 characters.'")
print(f"2**60 = {2**60}")
print()

for val in test_cases:
    result = format_bytes(val)
    is_valid = len(result) <= 10
    status = "✓" if is_valid else "✗ VIOLATION"
    print(f"Input: {val:22d} (< 2**60: {val < 2**60})")
    print(f"Output: '{result}' (length: {len(result)}) {status}")
    print()