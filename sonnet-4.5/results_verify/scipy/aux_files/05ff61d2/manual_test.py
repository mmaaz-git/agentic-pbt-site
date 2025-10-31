from io import BytesIO
import numpy as np
from scipy.io.matlab import loadmat, savemat
import warnings

bio = BytesIO()
data = {'0': np.array([1, 2, 3])}

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    savemat(bio, data)
    print(f"Warnings issued: {len(w)}")

bio.seek(0)
loaded = loadmat(bio)
print(f"'0' in loaded: {'0' in loaded}")
print(f"Loaded data keys: {list(loaded.keys())}")

# Let's also test with other digit-prefixed keys
print("\nTesting with '1test' key:")
bio2 = BytesIO()
data2 = {'1test': np.array([4, 5, 6])}

with warnings.catch_warnings(record=True) as w2:
    warnings.simplefilter("always")
    savemat(bio2, data2)
    print(f"Warnings issued: {len(w2)}")

bio2.seek(0)
loaded2 = loadmat(bio2)
print(f"'1test' in loaded: {'1test' in loaded2}")
print(f"Loaded data keys: {list(loaded2.keys())}")