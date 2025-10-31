import numpy as np

# Let's verify the strings are actually different
str1 = ''
str2 = ''
str3 = '\x00'

print("String representations:")
print(f"str1: {repr(str1)} (len={len(str1)}, bytes={str1.encode('utf-8')})")
print(f"str2: {repr(str2)} (len={len(str2)}, bytes={str2.encode('utf-8')})")
print(f"str3: {repr(str3)} (len={len(str3)}, bytes={str3.encode('utf-8')})")

print("\nComparisons:")
print(f"str1 == str2: {str1 == str2}")
print(f"str1 == str3: {str1 == str3}")
print(f"str2 == str3: {str2 == str3}")

print("\nOrd values:")
if len(str1) > 0:
    print(f"ord(str1[0]): {ord(str1[0])}")
if len(str2) > 0:
    print(f"ord(str2[0]): {ord(str2[0])}")
if len(str3) > 0:
    print(f"ord(str3[0]): {ord(str3[0])}")