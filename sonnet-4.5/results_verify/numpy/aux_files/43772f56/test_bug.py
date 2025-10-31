import numpy as np
import numpy.char as nc

print("Testing numpy.char case conversion functions:")
print("=" * 60)

# Test upper()
print("\nTesting upper():")
arr = np.array(['ß'])
print(f"upper('ß'):    numpy={nc.upper(arr)[0]!r}, python={'ß'.upper()!r}")

arr = np.array(['ﬁ'])
print(f"upper('ﬁ'):    numpy={nc.upper(arr)[0]!r}, python={'ﬁ'.upper()!r}")

arr = np.array(['ﬂ'])
print(f"upper('ﬂ'):    numpy={nc.upper(arr)[0]!r}, python={'ﬂ'.upper()!r}")

arr = np.array(['ﬀ'])
print(f"upper('ﬀ'):    numpy={nc.upper(arr)[0]!r}, python={'ﬀ'.upper()!r}")

arr = np.array(['ﬃ'])
print(f"upper('ﬃ'):    numpy={nc.upper(arr)[0]!r}, python={'ﬃ'.upper()!r}")

arr = np.array(['ﬄ'])
print(f"upper('ﬄ'):    numpy={nc.upper(arr)[0]!r}, python={'ﬄ'.upper()!r}")

arr = np.array(['ﬅ'])
print(f"upper('ﬅ'):    numpy={nc.upper(arr)[0]!r}, python={'ﬅ'.upper()!r}")

arr = np.array(['ﬆ'])
print(f"upper('ﬆ'):    numpy={nc.upper(arr)[0]!r}, python={'ﬆ'.upper()!r}")

# Test lower()
print("\nTesting lower():")
arr = np.array(['İ'])
print(f"lower('İ'):    numpy={nc.lower(arr)[0]!r}, python={'İ'.lower()!r}")

# Test swapcase()
print("\nTesting swapcase():")
arr = np.array(['ß'])
print(f"swapcase('ß'): numpy={nc.swapcase(arr)[0]!r}, python={'ß'.swapcase()!r}")

arr = np.array(['ﬁ'])
print(f"swapcase('ﬁ'): numpy={nc.swapcase(arr)[0]!r}, python={'ﬁ'.swapcase()!r}")

# Check array dtypes
print("\n" + "=" * 60)
print("Checking array dtypes:")
arr = np.array(['ß'])
print(f"Original array dtype: {arr.dtype}")
upper_arr = nc.upper(arr)
print(f"After upper() dtype: {upper_arr.dtype}")
print(f"Length of original: {len(arr[0])}")
print(f"Length after upper: {len(upper_arr[0])}")