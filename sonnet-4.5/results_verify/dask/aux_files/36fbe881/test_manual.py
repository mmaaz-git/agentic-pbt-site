from dask.widgets.widgets import format_bytes

n = 1_125_894_277_343_089_729
result = format_bytes(n)

print(f"n = {n}")
print(f"n < 2**60? {n < 2**60}")
print(f"format_bytes({n}) = {result!r}")
print(f"Length: {len(result)}")

assert n < 2**60
assert len(result) == 11
print("\nAll assertions passed!")