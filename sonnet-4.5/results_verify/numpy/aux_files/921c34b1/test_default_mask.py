import numpy as np
import numpy.ma as ma

# Test default mask behavior
print("Creating masked arrays without explicit mask:")
arr1 = ma.masked_array([1, 2, 3])
print(f"arr1.mask = {arr1.mask}, type = {type(arr1.mask)}")

arr2 = ma.masked_array([])
print(f"arr2 (empty).mask = {arr2.mask}, type = {type(arr2.mask)}")

print("\nChecking nomask:")
print(f"ma.nomask = {ma.nomask}, type = {type(ma.nomask)}")
print(f"arr1.mask is ma.nomask: {arr1.mask is ma.nomask}")
print(f"arr2.mask is ma.nomask: {arr2.mask is ma.nomask}")

# Test concatenation with nomask arrays
print("\nConcatenating arrays with nomask:")
arr3 = ma.masked_array([4, 5])
result = ma.concatenate([arr1, arr3])
print(f"result.mask = {result.mask}, type = {type(result.mask)}")
print(f"result.mask is ma.nomask: {result.mask is ma.nomask}")

# Test concatenating empty arrays without explicit mask
print("\nConcatenating empty arrays without explicit mask:")
arr4 = ma.masked_array([])
arr5 = ma.masked_array([])
result2 = ma.concatenate([arr4, arr5])
print(f"result2.mask = {result2.mask}, type = {type(result2.mask)}")
print(f"result2.mask is ma.nomask: {result2.mask is ma.nomask}")