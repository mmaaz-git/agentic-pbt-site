import tempfile
import numpy as np
from scipy import io
import os

with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
    fname = f.name

try:
    empty_arr = np.array([])

    io.savemat(fname, {'x': empty_arr}, oned_as='row')
    loaded_row = io.loadmat(fname)
    print(f"oned_as='row': {empty_arr.shape} -> {loaded_row['x'].shape}")
    print(f"Expected: (1, 0), Got: {loaded_row['x'].shape}")

    io.savemat(fname, {'x': empty_arr}, oned_as='column')
    loaded_col = io.loadmat(fname)
    print(f"oned_as='column': {empty_arr.shape} -> {loaded_col['x'].shape}")
    print(f"Expected: (0, 1), Got: {loaded_col['x'].shape}")

    non_empty = np.array([1.0, 2.0, 3.0])
    io.savemat(fname, {'x': non_empty}, oned_as='row')
    loaded_ne = io.loadmat(fname)
    print(f"\nNon-empty with oned_as='row': {non_empty.shape} -> {loaded_ne['x'].shape}")
    print(f"Expected: (1, 3), Got: {loaded_ne['x'].shape} ✓")

    # Additional tests to understand the behavior better
    print("\n--- Additional tests ---")

    # Test with default oned_as
    io.savemat(fname, {'x': empty_arr})
    loaded_default = io.loadmat(fname)
    print(f"Empty with default oned_as: {empty_arr.shape} -> {loaded_default['x'].shape}")

    # Test what happens with explicitly 2D empty arrays
    empty_2d_row = np.array([]).reshape(1, 0)
    io.savemat(fname, {'x': empty_2d_row})
    loaded_2d_row = io.loadmat(fname)
    print(f"Explicit (1,0) array: {empty_2d_row.shape} -> {loaded_2d_row['x'].shape}")

    empty_2d_col = np.array([]).reshape(0, 1)
    io.savemat(fname, {'x': empty_2d_col})
    loaded_2d_col = io.loadmat(fname)
    print(f"Explicit (0,1) array: {empty_2d_col.shape} -> {loaded_2d_col['x'].shape}")

finally:
    if os.path.exists(fname):
        os.unlink(fname)