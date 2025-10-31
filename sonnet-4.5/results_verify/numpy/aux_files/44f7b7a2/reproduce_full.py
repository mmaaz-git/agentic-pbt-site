import numpy as np
import numpy.strings as ns

arr = np.array(['\x00'], dtype=np.str_)

print("Testing various string operations on null byte '\\x00':")
print("=" * 60)

print(f"upper: numpy={repr(ns.upper(arr)[0])}, python={repr('\x00'.upper())}")
print(f"lower: numpy={repr(ns.lower(arr)[0])}, python={repr('\x00'.lower())}")
print(f"capitalize: numpy={repr(ns.capitalize(arr)[0])}, python={repr('\x00'.capitalize())}")
print(f"title: numpy={repr(ns.title(arr)[0])}, python={repr('\x00'.title())}")
print(f"swapcase: numpy={repr(ns.swapcase(arr)[0])}, python={repr('\x00'.swapcase())}")
print(f"strip: numpy={repr(ns.strip(arr)[0])}, python={repr('\x00'.strip())}")

left, mid, right = ns.partition(arr, 'X')
print(f"partition: numpy=({repr(left[0])}, {repr(mid[0])}, {repr(right[0])}), python={repr('\x00'.partition('X'))}")

print("\n" + "=" * 60)
print("Testing null bytes in the middle of strings:")
print("=" * 60)

# Test with null byte in middle
arr2 = np.array(['hel\x00lo'], dtype=np.str_)
print(f"'hel\\x00lo'.upper(): numpy={repr(ns.upper(arr2)[0])}, python={repr('hel\x00lo'.upper())}")

# Test length preservation
print("\n" + "=" * 60)
print("Testing length preservation:")
print(f"len('\\x00') = {len('\x00')}")
print(f"len(numpy result for '\\x00'.upper()) = {len(ns.upper(arr)[0])}")