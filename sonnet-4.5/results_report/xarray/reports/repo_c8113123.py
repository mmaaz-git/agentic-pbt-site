from xarray.backends.chunks import build_grid_chunks

# Test case that demonstrates the bug
result = build_grid_chunks(size=1, chunk_size=2)
print(f"Result: {result}")
print(f"Sum of chunks: {sum(result)}")
print(f"Expected sum: 1")

# This should be True but isn't
print(f"Sum equals size: {sum(result) == 1}")

# This assertion will fail
assert sum(result) == 1, f"Chunks sum to {sum(result)} instead of 1"