import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from xarray.core.formatting_html import _load_static_files

# First call to get the static files
result1 = _load_static_files()
print(f"First call returns type: {type(result1)}")
print(f"First call, first element (first 50 chars): {result1[0][:50]}...")

# Save the original content to verify later
original_content = result1[0]

# Mutate the cached list
result1[0] = "MUTATED_CONTENT"
print(f"\nMutated first element of the returned list")

# Second call should return the same cached list (now mutated)
result2 = _load_static_files()
print(f"\nSecond call returns type: {type(result2)}")
print(f"Second call, first element: {result2[0]}")

# Verify the bug exists
print(f"\nBug confirmed: Cache returns mutated value: {result2[0] == 'MUTATED_CONTENT'}")
print(f"Original content lost: {result2[0] != original_content}")