#!/usr/bin/env python3
"""Check what type nbytes returns for different numpy arrays"""

import numpy as np

# Test various numpy arrays to see what type nbytes returns
arrays = [
    ("Regular array", np.ones(100, dtype='f8')),
    ("Broadcasted array", np.broadcast_to(1, (100, 100))),
    ("Zero array", np.zeros((10, 10))),
    ("Strided array", np.ones((100, 100))[::2, ::2]),
    ("Transposed array", np.ones((10, 20)).T),
    ("View", np.ones(100).view()),
]

print("Testing nbytes return types:")
print("=" * 60)

for name, arr in arrays:
    nbytes = arr.nbytes
    print(f"{name:20} | nbytes type: {type(nbytes).__name__:15} | value: {nbytes}")
    print(f"                     | is int: {type(nbytes) is int:5} | isinstance(int): {isinstance(nbytes, int)}")
    print("-" * 60)

# Also check numpy integer types
print("\nNumPy integer types:")
print(f"np.intp: {np.intp}")
print(f"np.int_: {np.int_}")
print(f"np.int64: {np.int64}")

# Check if np.intp is different from int
test_intp = np.intp(10)
test_int = int(10)
print(f"\nnp.intp(10) type: {type(test_intp)}, is int: {type(test_intp) is int}")
print(f"int(10) type: {type(test_int)}, is int: {type(test_int) is int}")

# Check behavior in older numpy API
print("\n" + "=" * 60)
print("Simulating the bug scenario:")

# This simulates what happens in dask/sizeof.py lines 137-140
def sizeof_numpy_ndarray(x):
    if 0 in x.strides:
        xs = x[tuple(slice(None) if s != 0 else slice(1) for s in x.strides)]
        return xs.nbytes  # Line 139 - bug: doesn't wrap with int()
    return int(x.nbytes)  # Line 140 - wraps with int()

arr_broadcast = np.broadcast_to(1, (100, 100))
result = sizeof_numpy_ndarray(arr_broadcast)
print(f"Result type: {type(result)}, is Python int: {type(result) is int}")