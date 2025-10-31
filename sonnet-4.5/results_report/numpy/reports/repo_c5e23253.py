import numpy as np

# Create a 4x4 array
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]])

# Create a non-contiguous view by slicing every other row and column
non_contiguous = arr[::2, ::2]

print(f"Original array shape: {arr.shape}")
print(f"Non-contiguous view shape: {non_contiguous.shape}")
print(f"Non-contiguous view:\n{non_contiguous}")
print(f"Is input contiguous? {non_contiguous.flags.contiguous}")
print(f"Is input C-contiguous? {non_contiguous.flags.c_contiguous}")
print(f"Is input F-contiguous? {non_contiguous.flags.f_contiguous}")

# Create a matrix from the non-contiguous array with copy=False
m = np.matrix(non_contiguous, copy=False)

print(f"\nMatrix created with copy=False:")
print(f"Matrix:\n{m}")
print(f"Is matrix contiguous? {m.flags.contiguous}")
print(f"Is matrix C-contiguous? {m.flags.c_contiguous}")
print(f"Is matrix F-contiguous? {m.flags.f_contiguous}")

# Demonstrate the logic error
print("\n--- Logic Error Analysis ---")
order = 'C'  # This is always set to 'C' or 'F'
print(f"order = '{order}'")
print(f"bool(order) = {bool(order)}")
print(f"If order='C': (order or arr.flags.contiguous) = {bool(order or non_contiguous.flags.contiguous)}")
print(f"Therefore: not (order or arr.flags.contiguous) = {not (order or non_contiguous.flags.contiguous)}")
print("The condition is ALWAYS False, so line 167 (arr = arr.copy()) is NEVER executed!")