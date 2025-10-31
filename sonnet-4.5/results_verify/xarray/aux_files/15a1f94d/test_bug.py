#!/usr/bin/env python3
"""Test the reported bug in xarray.backends.chunks.build_grid_chunks"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from xarray.backends.chunks import build_grid_chunks

# Test case from bug report
size = 1
chunk_size = 2

chunks = build_grid_chunks(size, chunk_size)
print(f"size={size}, chunk_size={chunk_size}")
print(f"chunks={chunks}")
print(f"sum(chunks)={sum(chunks)}")
print(f"Expected sum: {size}")
print(f"Bug confirmed: {sum(chunks) != size}")