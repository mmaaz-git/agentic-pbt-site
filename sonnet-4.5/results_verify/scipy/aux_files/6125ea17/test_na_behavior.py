import pandas as pd
import numpy as np

print("Testing NA behavior with different operations:")
print("="*50)

# Test scalar NA operations
print("Scalar NA operations:")
print(f"pd.NA * 0 = {pd.NA * 0}")
print(f"pd.NA ** 0 = {pd.NA ** 0}")
print(f"0 * pd.NA = {0 * pd.NA}")
print(f"0 ** pd.NA = {0 ** pd.NA}")
print()

# Test with arrays
print("Array operations:")
na_float = pd.array([None, 1.0, 2.0], dtype="Float64")
print(f"Array: {na_float}")
print(f"Array * 0 = {na_float * 0}")
print(f"Array ** 0 = {na_float ** 0}")
print()

# Test mathematical identities
print("Mathematical identities:")
print(f"pd.NA + 0 = {pd.NA + 0}")
print(f"pd.NA - 0 = {pd.NA - 0}")
print(f"pd.NA * 1 = {pd.NA * 1}")
print(f"pd.NA / 1 = {pd.NA / 1}")
print(f"pd.NA ** 1 = {pd.NA ** 1}")
print(f"1 ** pd.NA = {1 ** pd.NA}")
print()

# Test with numpy
print("NumPy compatibility:")
print(f"np.multiply(pd.NA, 0) = {np.multiply(pd.NA, 0)}")
print(f"np.power(pd.NA, 0) = {np.power(pd.NA, 0)}")