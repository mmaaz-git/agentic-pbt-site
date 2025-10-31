import numpy as np

# Test numpy's behavior with empty arrays
empty_array = np.array([])
print(f"Empty array: {empty_array}")
print(f"Shape: {empty_array.shape}")

rolled = np.roll(empty_array, 1)
print(f"Rolled empty array: {rolled}")
print(f"Shape after roll: {rolled.shape}")

# Test with 2D empty array
empty_2d = np.array([]).reshape(0, 5)
print(f"\nEmpty 2D array shape: {empty_2d.shape}")
rolled_2d = np.roll(empty_2d, 1, axis=0)
print(f"Rolled 2D empty array shape: {rolled_2d.shape}")

# Test pandas behavior with empty index
import pandas as pd
empty_idx = pd.Index([])
print(f"\nEmpty pandas Index: {empty_idx}")
print(f"Length: {len(empty_idx)}")

# Since pandas Index doesn't have a roll method, let's simulate what it would do
if len(empty_idx) > 0:
    shift = 1 % len(empty_idx)
    if shift != 0:
        rolled_idx = empty_idx[-shift:].append(empty_idx[:-shift])
    else:
        rolled_idx = empty_idx[:]
else:
    # Handle empty case
    rolled_idx = empty_idx[:]

print(f"Rolled empty pandas Index: {rolled_idx}")