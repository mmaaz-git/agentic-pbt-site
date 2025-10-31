from dask.widgets.widgets import format_bytes

# Test edge cases around 1000 PiB
test_values = [
    2**50 * 999.99,  # Just under 1000 PiB
    2**50 * 1000,     # Exactly 1000 PiB
    2**50 * 1023,     # Maximum before rolling over
    2**60 - 1,        # Maximum value that should work according to docs
]

print("Testing edge cases:")
print("-" * 60)
for n in test_values:
    n_int = int(n)
    result = format_bytes(n_int)
    print(f"n = {n_int:,}")
    print(f"n / 2**50 = {n_int / 2**50:.2f}")
    print(f"n < 2**60? {n_int < 2**60}")
    print(f"format_bytes(n) = {result!r}")
    print(f"Length: {len(result)} chars")
    if len(result) > 10:
        print(f"⚠️  VIOLATES 10-character guarantee!")
    print("-" * 60)