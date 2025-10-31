import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env')

from dask.utils import format_bytes

n = 1125894277343089729
result = format_bytes(n)

print(f"n = {n}")
print(f"n < 2**60 = {n < 2**60}")
print(f"format_bytes({n}) = '{result}'")
print(f"Length of result = {len(result)}")

# Assertions from bug report
assert n < 2**60
assert len(result) == 11
assert result == '1000.00 PiB'

print("\nAll assertions passed - bug confirmed!")