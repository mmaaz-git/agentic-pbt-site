#!/usr/bin/env python3
"""Test numpy's handling of null bytes with dtype=bytes"""

import numpy as np

print("Testing numpy's dtype=bytes behavior with null bytes")
print("="*60)

# Test 1: Single null byte
print("\nTest 1: Single null byte")
data1 = [b'\x00']
arr1_bytes = np.array(data1, dtype=bytes)
arr1_object = np.array(data1, dtype=object)
print(f"Input: {data1}")
print(f"dtype=bytes result: {arr1_bytes}, dtype: {arr1_bytes.dtype}, value: {repr(arr1_bytes[0])}")
print(f"dtype=object result: {arr1_object}, dtype: {arr1_object.dtype}, value: {repr(arr1_object[0])}")

# Test 2: String starting with null
print("\nTest 2: String starting with null")
data2 = [b'\x00hello']
arr2_bytes = np.array(data2, dtype=bytes)
arr2_object = np.array(data2, dtype=object)
print(f"Input: {data2}")
print(f"dtype=bytes result: {arr2_bytes}, dtype: {arr2_bytes.dtype}, value: {repr(arr2_bytes[0])}")
print(f"dtype=object result: {arr2_object}, dtype: {arr2_object.dtype}, value: {repr(arr2_object[0])}")

# Test 3: String with null in middle
print("\nTest 3: String with null in middle")
data3 = [b'hello\x00world']
arr3_bytes = np.array(data3, dtype=bytes)
arr3_object = np.array(data3, dtype=object)
print(f"Input: {data3}")
print(f"dtype=bytes result: {arr3_bytes}, dtype: {arr3_bytes.dtype}, value: {repr(arr3_bytes[0])}")
print(f"dtype=object result: {arr3_object}, dtype: {arr3_object.dtype}, value: {repr(arr3_object[0])}")

# Test 4: Multiple null bytes
print("\nTest 4: Multiple null bytes")
data4 = [b'\x00\x00\x00']
arr4_bytes = np.array(data4, dtype=bytes)
arr4_object = np.array(data4, dtype=object)
print(f"Input: {data4}")
print(f"dtype=bytes result: {arr4_bytes}, dtype: {arr4_bytes.dtype}, value: {repr(arr4_bytes[0])}")
print(f"dtype=object result: {arr4_object}, dtype: {arr4_object.dtype}, value: {repr(arr4_object[0])}")

# Test 5: Normal string
print("\nTest 5: Normal string (control)")
data5 = [b'hello world']
arr5_bytes = np.array(data5, dtype=bytes)
arr5_object = np.array(data5, dtype=object)
print(f"Input: {data5}")
print(f"dtype=bytes result: {arr5_bytes}, dtype: {arr5_bytes.dtype}, value: {repr(arr5_bytes[0])}")
print(f"dtype=object result: {arr5_object}, dtype: {arr5_object.dtype}, value: {repr(arr5_object[0])}")

print("\n" + "="*60)
print("Key observation: dtype=bytes truncates at null bytes!")
print("dtype=bytes creates fixed-width S dtype where null = terminator")
print("dtype=object preserves the exact byte string")