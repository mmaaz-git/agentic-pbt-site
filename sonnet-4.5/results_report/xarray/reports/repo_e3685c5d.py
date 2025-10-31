from xarray.backends.chunks import build_grid_chunks

# Test the failing case: size=1, chunk_size=2
result = build_grid_chunks(size=1, chunk_size=2, region=None)
print(f"build_grid_chunks(size=1, chunk_size=2, region=None)")
print(f"Result: {result}")
print(f"Sum of chunks: {sum(result)}")
print(f"Expected sum: 1")
print(f"Bug: The chunks sum to {sum(result)} instead of 1")
print()

# Additional test cases to show the pattern
test_cases = [
    (1, 1),   # Should work correctly
    (1, 10),  # Should fail
    (2, 3),   # Should fail
    (3, 10),  # Should fail
    (5, 5),   # Should work correctly
    (10, 3),  # Should work correctly
]

print("Additional test cases:")
for size, chunk_size in test_cases:
    chunks = build_grid_chunks(size=size, chunk_size=chunk_size, region=None)
    chunks_sum = sum(chunks)
    status = "✓" if chunks_sum == size else "✗"
    print(f"size={size:2}, chunk_size={chunk_size:2}: chunks={chunks}, sum={chunks_sum:2}, expected={size:2} {status}")