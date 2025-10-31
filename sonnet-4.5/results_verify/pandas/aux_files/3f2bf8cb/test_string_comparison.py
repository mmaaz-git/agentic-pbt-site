import numpy as np

# Test if '' and '\x00' are truly different
empty_str = ''
null_byte = '\x00'

print(f"empty_str value: {repr(empty_str)}")
print(f"null_byte value: {repr(null_byte)}")
print(f"Are they equal? {empty_str == null_byte}")
print(f"empty_str length: {len(empty_str)}")
print(f"null_byte length: {len(null_byte)}")
print(f"empty_str bytes: {empty_str.encode('utf-8')}")
print(f"null_byte bytes: {null_byte.encode('utf-8')}")

# Test numpy handling
arr1 = np.array(['', '\x00'], dtype=object)
print(f"\nNumpy array: {arr1}")
print(f"Array[0] == Array[1]: {arr1[0] == arr1[1]}")
print(f"Array[0] repr: {repr(arr1[0])}")
print(f"Array[1] repr: {repr(arr1[1])}")

# Test with more null bytes and special chars
special_strings = ['', '\x00', '\x01', '\x00\x00', ' ']
from pandas import factorize

codes, cats = factorize(special_strings, sort=False)
print(f"\nTesting various strings: {[repr(s) for s in special_strings]}")
print(f"Factorize codes: {codes}")
print(f"Factorize categories: {[repr(c) for c in cats]}")