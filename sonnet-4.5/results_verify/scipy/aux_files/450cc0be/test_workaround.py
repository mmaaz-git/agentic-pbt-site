import numpy as np
import scipy.ndimage as ndi

input_array = np.ones((5, 5), dtype=bool)

# Test with border_value=1 as suggested workaround
closed_fixed = ndi.binary_closing(input_array, border_value=1)

print("Input (all True):")
print(input_array.astype(int))

print("\nClosed with border_value=1:")
print(closed_fixed.astype(int))

print(f"\nExtensiveness check: {np.all(input_array <= closed_fixed)}")
print(f"Pixels changed: {np.sum(closed_fixed) - np.sum(input_array)}")