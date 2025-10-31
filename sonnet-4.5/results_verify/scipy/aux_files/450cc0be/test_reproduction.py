import numpy as np
import scipy.ndimage as ndi

input_array = np.ones((5, 5), dtype=bool)

closed = ndi.binary_closing(input_array)

print("Input (all True):")
print(input_array.astype(int))

print("\nClosed:")
print(closed.astype(int))

print(f"\nExtensiveness check: {np.all(input_array <= closed)}")
print(f"Closing REMOVED {np.sum(input_array) - np.sum(closed)} pixels")