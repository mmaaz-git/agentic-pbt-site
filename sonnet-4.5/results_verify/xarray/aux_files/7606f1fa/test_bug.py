import numpy as np
from xarray.namedarray.core import NamedArray

print("Testing permute_dims with missing_dims='ignore'")
print("-" * 50)

data = np.arange(6).reshape(2, 3)
arr = NamedArray(('x', 'y'), data)

print(f"Original array dims: {arr.dims}")
print(f"Original array shape: {arr.shape}")

try:
    result = arr.permute_dims('x', 'z', missing_dims='ignore')
    print(f"Result dims after permute_dims('x', 'z', missing_dims='ignore'): {result.dims}")
except ValueError as e:
    print(f"ERROR: ValueError raised despite missing_dims='ignore'")
    print(f"Error message: {e}")

print("\nTesting with missing_dims='warn'")
print("-" * 50)

try:
    result = arr.permute_dims('x', 'z', missing_dims='warn')
    print(f"Result dims after permute_dims('x', 'z', missing_dims='warn'): {result.dims}")
except ValueError as e:
    print(f"ERROR: ValueError raised despite missing_dims='warn'")
    print(f"Error message: {e}")

print("\nTesting with missing_dims='raise' (default)")
print("-" * 50)

try:
    result = arr.permute_dims('x', 'z', missing_dims='raise')
    print(f"Result dims after permute_dims('x', 'z', missing_dims='raise'): {result.dims}")
except ValueError as e:
    print(f"EXPECTED: ValueError raised with missing_dims='raise'")
    print(f"Error message: {e}")