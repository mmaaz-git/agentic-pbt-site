from dask.utils import format_bytes

# Test the boundary case where format_bytes produces 11 characters
n = 1_125_899_906_842_624_000  # This equals 1000 * 2^50
result = format_bytes(n)

print(f"n = {n}")
print(f"n < 2**60 = {n < 2**60}")
print(f"result = {result!r}")
print(f"len(result) = {len(result)}")

# Verify the violation
assert n < 2**60, "Value should be less than 2^60"
assert len(result) == 11, f"Expected length 11, got {len(result)}"
assert result == '1000.00 PiB', f"Expected '1000.00 PiB', got {result!r}"

print("\nContract violation confirmed:")
print(f"  Documentation claims: 'For all values < 2**60, the output is always <= 10 characters'")
print(f"  Actual output: {result!r} has {len(result)} characters")