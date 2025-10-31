import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from xarray.backends.chunks import build_grid_chunks

# Test case from the bug report: size=0, chunk_size=5
result = build_grid_chunks(size=0, chunk_size=5, region=None)
print(f"Test: build_grid_chunks(size=0, chunk_size=5, region=None)")
print(f"Result: {result}")
print(f"Sum of chunks: {sum(result)}")
print(f"Expected sum: 0")
print(f"Bug: sum(chunks)={sum(result)} != size=0")