import numpy as np

# Test numpy's argmin/argmax behavior with all equal values
print("NumPy behavior for arrays with all equal values:\n")

# Single element
arr1 = np.array([0])
print(f"np.array([0]).argmin() = {arr1.argmin()}")
print(f"np.array([0]).argmax() = {arr1.argmax()}")

# Multiple equal elements
arr2 = np.array([5, 5, 5, 5])
print(f"np.array([5, 5, 5, 5]).argmin() = {arr2.argmin()}")
print(f"np.array([5, 5, 5, 5]).argmax() = {arr2.argmax()}")

# All zeros
arr3 = np.array([0, 0, 0])
print(f"np.array([0, 0, 0]).argmin() = {arr3.argmin()}")
print(f"np.array([0, 0, 0]).argmax() = {arr3.argmax()}")

# All negative
arr4 = np.array([-1, -1, -1])
print(f"np.array([-1, -1, -1]).argmin() = {arr4.argmin()}")
print(f"np.array([-1, -1, -1]).argmax() = {arr4.argmax()}")

print("\nNumPy documentation for argmin/argmax behavior:")
print("When there are multiple minimum/maximum values,")
print("the indices corresponding to the first occurrence are returned.")