#!/usr/bin/env python3
"""Trace through the build_grid_chunks function to understand the bug."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

def build_grid_chunks_traced(size, chunk_size, region=None):
    """Traced version of build_grid_chunks to see what's happening."""
    if region is None:
        region = slice(0, size)

    region_start = region.start or 0
    print(f"  region_start = {region_start}")

    # Generate the zarr chunks inside the region of this dim
    first_chunk = chunk_size - (region_start % chunk_size)
    print(f"  first_chunk = chunk_size - (region_start % chunk_size)")
    print(f"  first_chunk = {chunk_size} - ({region_start} % {chunk_size})")
    print(f"  first_chunk = {chunk_size} - {region_start % chunk_size}")
    print(f"  first_chunk = {first_chunk}")

    chunks_on_region = [first_chunk]
    print(f"  chunks_on_region = [{first_chunk}]")

    remaining = size - chunks_on_region[0]
    print(f"  remaining after first chunk = {size} - {chunks_on_region[0]} = {remaining}")

    full_chunks_count = remaining // chunk_size
    print(f"  full_chunks_count = {remaining} // {chunk_size} = {full_chunks_count}")

    chunks_on_region.extend([chunk_size] * full_chunks_count)
    print(f"  After adding full chunks: {chunks_on_region}")

    if remaining % chunk_size != 0:
        last_chunk = remaining % chunk_size
        print(f"  Adding last chunk: {last_chunk}")
        chunks_on_region.append(last_chunk)

    print(f"  Final chunks: {chunks_on_region}")
    print(f"  Sum: {sum(chunks_on_region)}")

    return tuple(chunks_on_region)

# Test the failing case
print("Tracing size=1, chunk_size=2:")
result = build_grid_chunks_traced(size=1, chunk_size=2, region=None)
print(f"Result: {result}\n")

print("Tracing size=3, chunk_size=10:")
result = build_grid_chunks_traced(size=3, chunk_size=10, region=None)
print(f"Result: {result}\n")

print("Tracing size=10, chunk_size=3:")
result = build_grid_chunks_traced(size=10, chunk_size=3, region=None)
print(f"Result: {result}\n")

print("Tracing size=5, chunk_size=5:")
result = build_grid_chunks_traced(size=5, chunk_size=5, region=None)
print(f"Result: {result}\n")