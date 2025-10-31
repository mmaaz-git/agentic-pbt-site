#!/usr/bin/env python3
"""Test to reproduce the reported null character bug in xarray"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import numpy as np
import xarray.coding.strings as xr_strings

# Test case 1: Simple null character test
print("Test 1: Simple null character")
text = '\x00'
arr = np.array([text], dtype=object)
print(f"Original array: {arr}")
print(f"Original text: {repr(arr[0])}, length: {len(arr[0])}")

encoded = xr_strings.encode_string_array(arr, encoding='utf-8')
print(f"Encoded array: {encoded}")
print(f"Encoded dtype: {encoded.dtype}")
print(f"Encoded value: {repr(encoded[0])}")

decoded = xr_strings.decode_bytes_array(encoded, encoding='utf-8')
print(f"Decoded array: {decoded}")
print(f"Decoded text: {repr(decoded[0])}, length: {len(decoded[0])}")

print(f"\nRound-trip successful? {arr[0] == decoded[0]}")
print(f"Original == Decoded: {repr(arr[0])} == {repr(decoded[0])}")

# Test case 2: String with null character in middle
print("\n" + "="*50)
print("Test 2: String with null character in middle")
text2 = 'hello\x00world'
arr2 = np.array([text2], dtype=object)
print(f"Original text: {repr(arr2[0])}, length: {len(arr2[0])}")

encoded2 = xr_strings.encode_string_array(arr2, encoding='utf-8')
print(f"Encoded value: {repr(encoded2[0])}")

decoded2 = xr_strings.decode_bytes_array(encoded2, encoding='utf-8')
print(f"Decoded text: {repr(decoded2[0])}, length: {len(decoded2[0])}")

print(f"\nRound-trip successful? {arr2[0] == decoded2[0]}")
print(f"Original == Decoded: {repr(arr2[0])} == {repr(decoded2[0])}")

# Test case 3: Multiple null characters
print("\n" + "="*50)
print("Test 3: Multiple null characters")
text3 = '\x00\x00\x00'
arr3 = np.array([text3], dtype=object)
print(f"Original text: {repr(arr3[0])}, length: {len(arr3[0])}")

encoded3 = xr_strings.encode_string_array(arr3, encoding='utf-8')
print(f"Encoded value: {repr(encoded3[0])}")

decoded3 = xr_strings.decode_bytes_array(encoded3, encoding='utf-8')
print(f"Decoded text: {repr(decoded3[0])}, length: {len(decoded3[0])}")

print(f"\nRound-trip successful? {arr3[0] == decoded3[0]}")
print(f"Original == Decoded: {repr(arr3[0])} == {repr(decoded3[0])}")

# Test case 4: Normal string (no null characters)
print("\n" + "="*50)
print("Test 4: Normal string (control test)")
text4 = 'hello world'
arr4 = np.array([text4], dtype=object)
print(f"Original text: {repr(arr4[0])}, length: {len(arr4[0])}")

encoded4 = xr_strings.encode_string_array(arr4, encoding='utf-8')
print(f"Encoded value: {repr(encoded4[0])}")

decoded4 = xr_strings.decode_bytes_array(encoded4, encoding='utf-8')
print(f"Decoded text: {repr(decoded4[0])}, length: {len(decoded4[0])}")

print(f"\nRound-trip successful? {arr4[0] == decoded4[0]}")
print(f"Original == Decoded: {repr(arr4[0])} == {repr(decoded4[0])}")