import numpy as np
import scipy.ndimage as ndi

# Create an input array of all True values
input_array = np.ones((5, 5), dtype=bool)

# Apply binary_closing with default parameters
closed = ndi.binary_closing(input_array)

print("Input (all True):")
print(input_array.astype(int))

print("\nClosed result:")
print(closed.astype(int))

# Check if extensiveness property holds (input should be subset of closed)
extensiveness_check = np.all(input_array <= closed)
print(f"\nExtensiveness property (X ⊆ closing(X)): {extensiveness_check}")

# Count how many pixels were removed (should be 0 for proper closing)
pixels_removed = np.sum(input_array) - np.sum(closed)
print(f"Pixels removed by closing: {pixels_removed}")

if pixels_removed > 0:
    print(f"\nBUG: Closing REMOVED {pixels_removed} pixels instead of only adding or keeping them!")