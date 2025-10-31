#!/usr/bin/env python3
"""Test the proposed fix for build_grid_chunks"""

def build_grid_chunks_fixed(
    size: int,
    chunk_size: int,
    region: slice | None = None,
) -> tuple[int, ...]:
    if region is None:
        region = slice(0, size)

    region_start = region.start or 0
    # Generate the zarr chunks inside the region of this dim
    # FIX: Ensure first chunk doesn't exceed size
    first_chunk_size = min(size, chunk_size - (region_start % chunk_size))
    chunks_on_region = [first_chunk_size]
    chunks_on_region.extend([chunk_size] * ((size - chunks_on_region[0]) // chunk_size))
    if (size - chunks_on_region[0]) % chunk_size != 0:
        chunks_on_region.append((size - chunks_on_region[0]) % chunk_size)
    return tuple(chunks_on_region)

# Test cases that failed with original
test_cases = [
    (1, 2),   # size < chunk_size
    (1, 10),  # size much smaller than chunk_size
    (2, 3),   # size < chunk_size, both > 1
    (5, 5),   # size == chunk_size
    (10, 3),  # size > chunk_size, not divisible
    (12, 4),  # size > chunk_size, divisible
]

print("Testing fixed version:")
all_correct = True
for size, chunk_size in test_cases:
    chunks = build_grid_chunks_fixed(size, chunk_size)
    chunks_sum = sum(chunks)
    is_correct = chunks_sum == size
    print(f"size={size:3}, chunk_size={chunk_size:3}, chunks={chunks}, sum={chunks_sum:3}, correct={is_correct}")
    if not is_correct:
        all_correct = False

print(f"\nAll tests passed: {all_correct}")

# Test with regions too
print("\nTests with region parameter:")
for region_start in [0, 1, 2, 3]:
    size = 5
    chunk_size = 3
    region = slice(region_start, region_start + size)
    chunks = build_grid_chunks_fixed(size, chunk_size, region=region)
    chunks_sum = sum(chunks)
    is_correct = chunks_sum == size
    print(f"size={size}, chunk_size={chunk_size}, region={region}, chunks={chunks}, sum={chunks_sum}, correct={is_correct}")
    if not is_correct:
        all_correct = False

print(f"\nAll tests including regions passed: {all_correct}")