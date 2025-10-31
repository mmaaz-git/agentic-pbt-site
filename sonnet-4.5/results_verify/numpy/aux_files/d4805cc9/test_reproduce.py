import numpy as np
import numpy.strings as ns

print("Testing numpy.strings.index with mixed results:")
print("=" * 50)

arr = np.array(['0', ''])
sub = '0'

print(f"Input array: {repr(arr)}")
print(f"Substring to find: '{sub}'")
print()

# Test find function
find_result = ns.find(arr, sub)
print(f"find(arr, '0'): {find_result}")

# Test index function
try:
    index_result = ns.index(arr, sub)
    print(f"index(arr, '0'): {index_result}")
except ValueError as e:
    print(f"index(arr, '0'): raises ValueError: {e}")
    print("BUG: find succeeds and returns [0, -1], but index raises ValueError")

print()
print("Testing rindex as well:")
print("-" * 30)

# Test rfind function
rfind_result = ns.rfind(arr, sub)
print(f"rfind(arr, '0'): {rfind_result}")

# Test rindex function
try:
    rindex_result = ns.rindex(arr, sub)
    print(f"rindex(arr, '0'): {rindex_result}")
except ValueError as e:
    print(f"rindex(arr, '0'): raises ValueError: {e}")
    print("BUG: rfind succeeds and returns [0, -1], but rindex raises ValueError")