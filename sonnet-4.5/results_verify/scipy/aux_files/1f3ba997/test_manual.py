import numpy as np
import scipy.io.matlab as matlab
import tempfile
import os

arr_1d = np.array([1.0, 2.0, 3.0])
print(f"Original: shape={arr_1d.shape}, ndim={arr_1d.ndim}")

with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
    temp_filename = f.name
    matlab.savemat(f.name, {'data': arr_1d})
    loaded = matlab.loadmat(f.name)
    loaded_arr = loaded['data']

print(f"Loaded:   shape={loaded_arr.shape}, ndim={loaded_arr.ndim}")
print(f"Shapes match: {arr_1d.shape == loaded_arr.shape}")

# Clean up
os.unlink(temp_filename)

# Test with squeeze_me parameter
print("\n--- Testing with squeeze_me=True ---")
# Multi-element array
arr_multi = np.array([1.0, 2.0, 3.0])
print(f"Multi-element original: shape={arr_multi.shape}")

with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
    temp_filename = f.name
    matlab.savemat(f.name, {'data': arr_multi})
    loaded = matlab.loadmat(f.name, squeeze_me=True)
    loaded_arr = loaded['data']

print(f"Multi-element loaded with squeeze_me: shape={loaded_arr.shape}, type={type(loaded_arr)}")
os.unlink(temp_filename)

# Single-element array
arr_single = np.array([1.0])
print(f"\nSingle-element original: shape={arr_single.shape}")

with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
    temp_filename = f.name
    matlab.savemat(f.name, {'data': arr_single})
    loaded = matlab.loadmat(f.name, squeeze_me=True)
    loaded_arr = loaded['data']

print(f"Single-element loaded with squeeze_me: shape={getattr(loaded_arr, 'shape', 'N/A (scalar)')}, type={type(loaded_arr)}")
os.unlink(temp_filename)