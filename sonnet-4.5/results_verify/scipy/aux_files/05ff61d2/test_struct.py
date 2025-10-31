from io import BytesIO
import numpy as np
from scipy.io.matlab import loadmat, savemat
import warnings

# Test struct fields with digit-prefixed keys
print("Testing struct fields with digit-prefixed keys:")
bio = BytesIO()

# Create a struct with digit-prefixed field
struct_data = {'mystruct': {'0field': np.array([1, 2, 3]), 'valid_field': np.array([4, 5, 6])}}

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    savemat(bio, struct_data)
    print(f"Warnings issued: {len(w)}")
    if w:
        for warning in w:
            print(f"  Warning: {warning.message}")

bio.seek(0)
loaded = loadmat(bio)
print(f"Loaded keys: {list(loaded.keys())}")
if 'mystruct' in loaded:
    print(f"Struct fields: {loaded['mystruct'].dtype.names}")

# Test with underscore-prefixed key at top level
print("\nTesting underscore-prefixed top-level key:")
bio2 = BytesIO()
data2 = {'_hidden': np.array([1, 2, 3]), 'visible': np.array([4, 5, 6])}

with warnings.catch_warnings(record=True) as w2:
    warnings.simplefilter("always")
    savemat(bio2, data2)
    print(f"Warnings issued: {len(w2)}")
    if w2:
        for warning in w2:
            print(f"  Warning: {warning.message}")

bio2.seek(0)
loaded2 = loadmat(bio2)
print(f"Loaded keys: {list(loaded2.keys())}")
print(f"'_hidden' in loaded: {'_hidden' in loaded2}")
print(f"'visible' in loaded: {'visible' in loaded2}")