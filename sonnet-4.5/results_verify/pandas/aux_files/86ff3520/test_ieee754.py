import numpy as np

# Test IEEE 754 signed zero behavior
print("IEEE 754 signed zero behavior:")
print("="*50)

# Create positive and negative zeros
pos_zero = 0.0
neg_zero = -0.0

# Test equality
print(f"0.0 == -0.0: {pos_zero == neg_zero}")
print(f"np.equal(0.0, -0.0): {np.equal(pos_zero, neg_zero)}")

# Test array equality
arr1 = np.array([0.0])
arr2 = np.array([-0.0])
print(f"np.array_equal([0.0], [-0.0]): {np.array_equal(arr1, arr2)}")

# Test bit representations
print("\nBit representations:")
print(f"0.0 bits: {np.array([0.0]).view('u8')[0]:064b}")
print(f"-0.0 bits: {np.array([-0.0]).view('u8')[0]:064b}")

# Test Python's hash behavior
print("\nPython's hash behavior:")
print(f"hash(0.0): {hash(0.0)}")
print(f"hash(-0.0): {hash(-0.0)}")
print(f"hash(0.0) == hash(-0.0): {hash(0.0) == hash(-0.0)}")

# Test numpy array hash (though this is not officially supported)
print("\nNumpy behavior:")
arr_pos = np.array([0.0, 1.0, 2.0])
arr_neg = np.array([-0.0, 1.0, 2.0])
print(f"Arrays equal: {np.array_equal(arr_pos, arr_neg)}")
print(f"All close: {np.allclose(arr_pos, arr_neg)}")