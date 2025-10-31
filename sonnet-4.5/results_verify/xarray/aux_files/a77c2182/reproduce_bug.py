import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import numpy as np
from xarray.namedarray.core import NamedArray

print("Test case 1: dims=('dim_1',)")
data = np.array([1, 2])
arr = NamedArray(("dim_1",), data)
expanded = arr.expand_dims()
print(f"Original dims: {arr.dims}")
print(f"Expanded dims: {expanded.dims}")
print(f"Duplicate? {len(expanded.dims) != len(set(expanded.dims))}")
print()

print("Test case 2: dims=('dim_0', 'dim_2')")
data = np.ones((2, 3))
arr = NamedArray(("dim_0", "dim_2"), data)
expanded = arr.expand_dims()
print(f"Original dims: {arr.dims}")
print(f"Expanded dims: {expanded.dims}")
print(f"Duplicate? {len(expanded.dims) != len(set(expanded.dims))}")
print()

print("Test case 3: dims=('a', 'b', 'dim_3')")
data = np.ones((2, 3, 4))
arr = NamedArray(("a", "b", "dim_3"), data)
expanded = arr.expand_dims()
print(f"Original dims: {arr.dims}")
print(f"Expanded dims: {expanded.dims}")
print(f"Duplicate? {len(expanded.dims) != len(set(expanded.dims))}")
print()

print("Test case 4 (should work): dims=('x', 'y')")
data = np.ones((2, 3))
arr = NamedArray(("x", "y"), data)
expanded = arr.expand_dims()
print(f"Original dims: {arr.dims}")
print(f"Expanded dims: {expanded.dims}")
print(f"Duplicate? {len(expanded.dims) != len(set(expanded.dims))}")