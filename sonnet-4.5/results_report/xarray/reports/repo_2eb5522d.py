import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from xarray.core.formatting_html import _load_static_files

# Get initial cached result
result1 = _load_static_files()
print(f"Initial length: {len(result1)}")
print(f"Type of result: {type(result1)}")

# Mutate the cached list
result1.append("CORRUPTED DATA")
print(f"Length after appending to result1: {len(result1)}")

# Get the supposedly cached result again
result2 = _load_static_files()
print(f"Length of result2 (should be same as initial): {len(result2)}")
print(f"Last element of result2: {result2[-1]}")

# Check if the cache was corrupted
if len(result2) != 2:
    print("\nERROR: Cache was mutated! The cached list is not protected from modifications.")
    print(f"Expected length: 2, Got: {len(result2)}")
else:
    print("\nCache is properly protected.")