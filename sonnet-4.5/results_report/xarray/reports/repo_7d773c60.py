import numpy as np
from xarray.namedarray.core import NamedArray

# Create a simple NamedArray with 2 dimensions
data = np.arange(6).reshape(2, 3)
arr = NamedArray(('x', 'y'), data)

print("Original array dimensions:", arr.dims)
print("Original array shape:", arr.shape)
print()

# Try to permute dims with a missing dimension 'z' using missing_dims='ignore'
print("Attempting: arr.permute_dims('x', 'z', missing_dims='ignore')")
try:
    result = arr.permute_dims('x', 'z', missing_dims='ignore')
    print("Result dimensions:", result.dims)
    print("Result shape:", result.shape)
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")