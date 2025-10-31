#!/usr/bin/env python3
import numpy as np
from xarray.core import dtypes

# Test xarray's dtypes.isdtype wrapper
scalar = np.int32(5)
print(f"Testing xarray.core.dtypes.isdtype with scalar: {scalar}")
print(f"Type: {type(scalar)}")

try:
    result = dtypes.isdtype(scalar, 'integral')
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e.__class__.__name__}: {e}")

# Test with dtype
print(f"\nTesting with .dtype: {scalar.dtype}")
try:
    result = dtypes.isdtype(scalar.dtype, 'integral')
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")

# Check the wrapper's conditions
print(f"\nIs np.dtype: {isinstance(scalar, np.dtype)}")
print(f"Is np.dtype: {isinstance(scalar.dtype, np.dtype)}")