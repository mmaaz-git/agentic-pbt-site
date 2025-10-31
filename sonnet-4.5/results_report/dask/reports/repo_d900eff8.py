import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from dask.utils import format_bytes

n = 1_125_894_277_343_089_729
result = format_bytes(n)

print(f"format_bytes({n}) = '{result}'")
print(f"Length: {len(result)} characters")
print(f"Expected: <= 10 characters")
print(f"Violates documented invariant: {len(result) > 10}")