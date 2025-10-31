import numpy as np
from scipy import io
import tempfile
import os

# Let's understand what MATLAB itself expects
# Row vectors should be (1, n) and column vectors should be (n, 1)

print("Testing the documented behavior of oned_as parameter:\n")

# Test 1: Non-empty array behavior (documented)
arr = np.array([1, 2, 3])
print(f"Non-empty 1D array: shape {arr.shape}")

with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
    fname = f.name

try:
    # Save as row
    io.savemat(fname, {'x': arr}, oned_as='row')
    loaded = io.loadmat(fname)
    print(f"  oned_as='row' -> {loaded['x'].shape} (expected (1, 3))")

    # Save as column
    io.savemat(fname, {'x': arr}, oned_as='column')
    loaded = io.loadmat(fname)
    print(f"  oned_as='column' -> {loaded['x'].shape} (expected (3, 1))")

    print("\nEmpty 1D array: shape (0,)")
    empty = np.array([])

    # By the same logic, empty arrays should follow the pattern:
    # row: (0,) -> (1, 0) - one row, zero columns
    # column: (0,) -> (0, 1) - zero rows, one column

    io.savemat(fname, {'x': empty}, oned_as='row')
    loaded = io.loadmat(fname)
    print(f"  oned_as='row' -> {loaded['x'].shape} (logically should be (1, 0))")

    io.savemat(fname, {'x': empty}, oned_as='column')
    loaded = io.loadmat(fname)
    print(f"  oned_as='column' -> {loaded['x'].shape} (logically should be (0, 1))")

finally:
    if os.path.exists(fname):
        os.unlink(fname)

print("\nConclusion:")
print("The documentation states that oned_as controls how 1-D arrays are written:")
print("- 'row': write as row vectors")
print("- 'column': write as column vectors")
print("\nA row vector has shape (1, n) and a column vector has shape (m, 1).")
print("Therefore, an empty 1D array should become:")
print("- (1, 0) when oned_as='row'")
print("- (0, 1) when oned_as='column'")
print("\nThe current behavior (always (0, 0)) is inconsistent with the documented purpose.")