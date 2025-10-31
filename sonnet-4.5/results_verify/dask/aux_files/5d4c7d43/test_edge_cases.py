from dask.utils import format_bytes

# Test boundary cases for the 10-character guarantee
test_cases = [
    # Test values around 1000 PiB
    (999 * 2**50, "999.00 PiB"),
    (1000 * 2**50, "1000.00 PiB"),  # 11 chars - violates guarantee
    (1001 * 2**50, "1001.00 PiB"),  # 11 chars - violates guarantee
    (1023.99 * 2**50, "1023.99 PiB"),  # 11 chars - violates guarantee
    (1024 * 2**50, "1024.00 PiB"),  # 11 chars - violates guarantee

    # Test maximum value < 2**60
    (2**60 - 1, None),  # Let's see what this produces
]

print("Testing format_bytes edge cases:")
print("-" * 50)

for value, expected in test_cases:
    result = format_bytes(value)
    length = len(result)
    passes = length <= 10

    print(f"Value: {value}")
    print(f"  Result: '{result}'")
    print(f"  Length: {length}")
    print(f"  Pass 10-char test: {passes}")

    if expected:
        print(f"  Expected: '{expected}'")
        print(f"  Matches: {result == expected}")

    # Check the documented contract
    if value < 2**60 and length > 10:
        print(f"  *** VIOLATES CONTRACT: Value < 2**60 but output has {length} > 10 characters")

    print()

# Check that the values are indeed < 2**60
print(f"2**60 = {2**60}")
print(f"All test values < 2**60: {all(v < 2**60 for v, _ in test_cases)}")