import dask

# Try creating a task graph with an empty dict value
dsk = {
    'x': {},  # Empty dict as literal
    'y': (dict, []),  # Empty dict as task
    'z': (dict, [['a', 1], ['b', 2]])  # Non-empty dict as task
}

# Process with unquote
from dask.diagnostics.profile_visualize import unquote

print("Testing unquote on different dict representations:")
print(f"Empty dict literal: {{}}")
print(f"unquote({{}}) = {unquote({})}")

print(f"\nEmpty dict task: (dict, [])")
try:
    result = unquote((dict, []))
    print(f"unquote((dict, [])) = {result}")
except IndexError as e:
    print(f"unquote((dict, [])) raises IndexError: {e}")

print(f"\nNon-empty dict task: (dict, [['a', 1], ['b', 2]])")
result = unquote((dict, [['a', 1], ['b', 2]]))
print(f"unquote((dict, [['a', 1], ['b', 2]])) = {result}")

# Test if (dict, []) can actually appear in real dask computations
print("\n\nTesting if (dict, []) appears in real dask computations:")
from dask import delayed

@delayed
def make_dict(items):
    return dict(items)

# Create a delayed computation with empty list
result = make_dict([])
print(f"Delayed task for dict([]): {result.key}: {result.dask[result.key]}")

# Also test with the profiler to see if such tasks could be encountered
from dask.diagnostics import Profiler
from dask.threaded import get

dsk2 = {
    'empty': (dict, []),
    'full': (dict, [['x', 1]])
}

print("\nTrying to execute task graph with (dict, []):")
try:
    with Profiler() as prof:
        result = get(dsk2, 'empty')
    print(f"Successfully executed! Result: {result}")
except Exception as e:
    print(f"Failed to execute: {e}")

print("\nTrying to execute task graph with (dict, [['x', 1]]):")
try:
    with Profiler() as prof:
        result = get(dsk2, 'full')
    print(f"Successfully executed! Result: {result}")
except Exception as e:
    print(f"Failed to execute: {e}")