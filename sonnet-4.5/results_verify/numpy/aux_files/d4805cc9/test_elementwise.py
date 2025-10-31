import numpy as np
import numpy.strings as ns

print("Testing element-wise behavior:")
print("=" * 60)

# Test 1: All elements contain substring
arr1 = np.array(['hello world', 'world peace', 'new world'])
print(f"Array 1: {repr(arr1)}")
print(f"find(arr1, 'world'): {ns.find(arr1, 'world')}")
try:
    print(f"index(arr1, 'world'): {ns.index(arr1, 'world')}")
except ValueError as e:
    print(f"index(arr1, 'world'): raises ValueError: {e}")

print()

# Test 2: No elements contain substring
arr2 = np.array(['hello', 'goodbye', 'welcome'])
print(f"Array 2: {repr(arr2)}")
print(f"find(arr2, 'world'): {ns.find(arr2, 'world')}")
try:
    print(f"index(arr2, 'world'): {ns.index(arr2, 'world')}")
except ValueError as e:
    print(f"index(arr2, 'world'): raises ValueError: {e}")

print()

# Test 3: Some elements contain substring (mixed case)
arr3 = np.array(['hello world', 'goodbye', 'new world'])
print(f"Array 3: {repr(arr3)}")
print(f"find(arr3, 'world'): {ns.find(arr3, 'world')}")
try:
    print(f"index(arr3, 'world'): {ns.index(arr3, 'world')}")
except ValueError as e:
    print(f"index(arr3, 'world'): raises ValueError: {e}")

print()

# Test 4: Single element array with substring
arr4 = np.array(['hello world'])
print(f"Array 4: {repr(arr4)}")
print(f"find(arr4, 'world'): {ns.find(arr4, 'world')}")
try:
    print(f"index(arr4, 'world'): {ns.index(arr4, 'world')}")
except ValueError as e:
    print(f"index(arr4, 'world'): raises ValueError: {e}")

print()

# Test 5: Single element array without substring
arr5 = np.array(['hello'])
print(f"Array 5: {repr(arr5)}")
print(f"find(arr5, 'world'): {ns.find(arr5, 'world')}")
try:
    print(f"index(arr5, 'world'): {ns.index(arr5, 'world')}")
except ValueError as e:
    print(f"index(arr5, 'world'): raises ValueError: {e}")