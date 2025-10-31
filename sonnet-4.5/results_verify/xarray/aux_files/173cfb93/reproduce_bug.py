import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from xarray.backends.chunks import build_grid_chunks

result = build_grid_chunks(size=0, chunk_size=5, region=None)
print(f"size=0, chunk_size=5")
print(f"Result: {result}")
print(f"Sum of chunks: {sum(result)}")
print(f"Expected: 0")
print(f"Bug: {sum(result)} != 0")
