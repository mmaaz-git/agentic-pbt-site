import numpy as np
from pandas.core.util.hashing import hash_array

# Test case 1: Short hash_key (5 bytes)
print("Test 1: Short hash_key")
try:
    arr = np.array(["test"], dtype=object)
    result = hash_array(arr, hash_key="short")
    print(f"Success: {result}")
except ValueError as e:
    print(f"ValueError: {e}")

print("\n" + "="*50 + "\n")

# Test case 2: Multi-byte UTF-8 that results in 19 bytes
print("Test 2: Multi-byte UTF-8 hash_key")
try:
    arr = np.array(["test"], dtype=object)
    result = hash_array(arr, hash_key="000000000000000🦄")
    print(f"Success: {result}")
except ValueError as e:
    print(f"ValueError: {e}")

print("\n" + "="*50 + "\n")

# Test case 3: Correct 16-byte hash_key (default behavior)
print("Test 3: Correct 16-byte hash_key (default)")
try:
    arr = np.array(["test"], dtype=object)
    result = hash_array(arr)  # Uses default "0123456789123456"
    print(f"Success with default: {result}")
except ValueError as e:
    print(f"ValueError: {e}")

print("\n" + "="*50 + "\n")

# Test case 4: Custom 16-byte hash_key
print("Test 4: Custom 16-byte hash_key")
try:
    arr = np.array(["test"], dtype=object)
    result = hash_array(arr, hash_key="abcdefghijklmnop")  # Exactly 16 bytes
    print(f"Success with custom 16-byte key: {result}")
except ValueError as e:
    print(f"ValueError: {e}")

print("\n" + "="*50 + "\n")

# Test case 5: Longer hash_key (20 bytes)
print("Test 5: Longer hash_key (20 bytes)")
try:
    arr = np.array(["test"], dtype=object)
    result = hash_array(arr, hash_key="12345678901234567890")  # 20 bytes
    print(f"Success: {result}")
except ValueError as e:
    print(f"ValueError: {e}")

print("\n" + "="*50 + "\n")

# Test what's the actual length of the multi-byte case
print("Test 6: Checking byte lengths")
test_key = "000000000000000🦄"
print(f"String: '{test_key}'")
print(f"Length as string: {len(test_key)}")
print(f"Length as UTF-8 bytes: {len(test_key.encode('utf-8'))}")
print(f"Encoded bytes: {test_key.encode('utf-8')}")