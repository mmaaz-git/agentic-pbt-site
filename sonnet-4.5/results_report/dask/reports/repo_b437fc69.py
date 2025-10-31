import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from dask.diagnostics.profile_visualize import unquote
from dask.core import istask

# Test case 1: Empty list argument to dict
task1 = (dict, [])
print(f"Bug 1 - Input: {task1}")
print(f"istask: {istask(task1)}")
try:
    result = unquote(task1)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print()

# Test case 2: List with empty sublist
task2 = (dict, [[]])
print(f"Bug 2 - Input: {task2}")
print(f"istask: {istask(task2)}")
try:
    result = unquote(task2)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")