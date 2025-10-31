import numpy as np
import scipy.ndimage as ndi

# Create an input array of all True values
input_array = np.ones((5, 5), dtype=bool)

print("Original input (all True):")
print(input_array.astype(int))

# Step 1: Dilation (with default border_value=0)
dilated = ndi.binary_dilation(input_array, border_value=0)
print("\nAfter dilation (border_value=0):")
print(dilated.astype(int))

# Step 2: Erosion of dilated result (with default border_value=0)
eroded = ndi.binary_erosion(dilated, border_value=0)
print("\nAfter erosion (border_value=0) - this is the closing result:")
print(eroded.astype(int))

print("\n--- Now with border_value=1 ---")

# Binary closing with border_value=1
closed_bv1 = ndi.binary_closing(input_array, border_value=1)
print("\nBinary closing with border_value=1:")
print(closed_bv1.astype(int))

# Check extensiveness with border_value=1
extensiveness_check_bv1 = np.all(input_array <= closed_bv1)
print(f"\nExtensiveness property with border_value=1: {extensiveness_check_bv1}")

# Compare with binary_opening
opened = ndi.binary_opening(input_array)
print("\n--- Comparison with binary_opening ---")
print("Binary opening result (default border_value=0):")
print(opened.astype(int))
idempotency_check = np.all(opened <= input_array)
print(f"Opening idempotency (opening(X) ⊆ X): {idempotency_check}")