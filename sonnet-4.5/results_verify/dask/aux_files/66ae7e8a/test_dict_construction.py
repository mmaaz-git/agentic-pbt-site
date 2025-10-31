from dask.core import istask

# Test various dict representations
empty_dict_repr = (dict, [])
print(f"Is (dict, []) a task? {istask(empty_dict_repr)}")

simple_dict_repr = (dict, [["a", 1]])
print(f"Is (dict, [['a', 1]]) a task? {istask(simple_dict_repr)}")

# Let's also test how dask normally creates such representations
import dask

# Create a simple dask computation involving empty dict
@dask.delayed
def return_empty_dict():
    return {}

@dask.delayed
def return_simple_dict():
    return {"a": 1}

# Get the task graph
empty_task = return_empty_dict()
simple_task = return_simple_dict()

print("\nDask graph for empty dict:", empty_task.dask)
print("\nDask graph for simple dict:", simple_task.dask)