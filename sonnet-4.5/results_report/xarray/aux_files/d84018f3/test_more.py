from xarray.backends.chunks import build_grid_chunks

# Test multiple cases where size < chunk_size
test_cases = [
    (1, 2),
    (1, 3),
    (2, 3),
    (5, 10),
    (10, 20),
    (99, 100),
]

print("Testing cases where size < chunk_size:")
print("-" * 50)
for size, chunk_size in test_cases:
    result = build_grid_chunks(size=size, chunk_size=chunk_size)
    sum_chunks = sum(result)
    correct = sum_chunks == size
    status = "✓" if correct else "✗ BUG"
    print(f"size={size:3}, chunk_size={chunk_size:3} -> chunks={result}, sum={sum_chunks:3} {status}")

print("\nTesting normal cases where size >= chunk_size:")
print("-" * 50)
normal_cases = [
    (10, 3),
    (100, 20),
    (50, 10),
]
for size, chunk_size in normal_cases:
    result = build_grid_chunks(size=size, chunk_size=chunk_size)
    sum_chunks = sum(result)
    correct = sum_chunks == size
    status = "✓" if correct else "✗ BUG"
    print(f"size={size:3}, chunk_size={chunk_size:3} -> chunks={result}, sum={sum_chunks:3} {status}")