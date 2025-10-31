#!/usr/bin/env python3
"""Test more edge cases for build_grid_chunks"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from xarray.backends.chunks import build_grid_chunks

test_cases = [
    (1, 2),   # size < chunk_size
    (1, 10),  # size much smaller than chunk_size
    (2, 3),   # size < chunk_size, both > 1
    (5, 5),   # size == chunk_size
    (10, 3),  # size > chunk_size, not divisible
    (12, 4),  # size > chunk_size, divisible
    (10, 1),  # chunk_size = 1
]

for size, chunk_size in test_cases:
    chunks = build_grid_chunks(size, chunk_size)
    chunks_sum = sum(chunks)
    is_correct = chunks_sum == size
    print(f"size={size:3}, chunk_size={chunk_size:3}, chunks={chunks}, sum={chunks_sum:3}, correct={is_correct}")

# Also test with region parameter
print("\nTests with region parameter:")
for region_start in [0, 1, 2, 3]:
    size = 5
    chunk_size = 3
    region = slice(region_start, region_start + size)
    chunks = build_grid_chunks(size, chunk_size, region=region)
    chunks_sum = sum(chunks)
    is_correct = chunks_sum == size
    print(f"size={size}, chunk_size={chunk_size}, region={region}, chunks={chunks}, sum={chunks_sum}, correct={is_correct}")