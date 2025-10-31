from xarray.backends.chunks import build_grid_chunks

# Test case that demonstrates the bug
size = 1
chunk_size = 2

print(f"Testing build_grid_chunks with size={size}, chunk_size={chunk_size}")
result = build_grid_chunks(size=size, chunk_size=chunk_size)
print(f"Result: {result}")
print(f"Sum of chunks: {sum(result)}")
print(f"Expected sum: {size}")
print(f"ERROR: Sum of chunks ({sum(result)}) != size ({size})")

# Verify the assertion fails
try:
    assert sum(result) == size
    print("Assertion passed (unexpected)")
except AssertionError:
    print("AssertionError: sum(result) != size")