from dask.utils import format_bytes

n = 1_125_894_277_343_089_729
result = format_bytes(n)

print(f"Input: {n}")
print(f"Input < 2**60: {n < 2**60}")
print(f"Output: {result!r}")
print(f"Output length: {len(result)}")

assert len(result) <= 10, f"Expected <= 10 characters, got {len(result)}"