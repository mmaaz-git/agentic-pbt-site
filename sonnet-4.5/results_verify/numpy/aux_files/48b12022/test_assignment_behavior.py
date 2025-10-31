import numpy as np

# Create a recarray with int64 dtype
dtype = np.dtype([('a', 'i8')])
rec = np.recarray((1,), dtype=dtype)

# Try assigning a uint64 value that exceeds int64 max
data = np.array([9_223_372_036_854_775_808], dtype=np.uint64)
print(f"Input data: {data[0]} (dtype: {data.dtype})")
print(f"Max int64: {np.iinfo(np.int64).max}")

# Direct assignment (what happens in fromarrays line 659)
rec['a'] = data
print(f"After assignment: {rec['a'][0]} (dtype: {rec['a'].dtype})")
print(f"Data corrupted: {data[0] != rec['a'][0]}")

# Test what happens with astype
print("\nTesting astype with different casting modes:")
try:
    safe_cast = data.astype(np.int64, casting='safe')
    print(f"Safe cast succeeded: {safe_cast}")
except TypeError as e:
    print(f"Safe cast failed: {e}")

unsafe_cast = data.astype(np.int64, casting='unsafe')
print(f"Unsafe cast result: {unsafe_cast}")

# Test what the default behavior is
default_cast = data.astype(np.int64)
print(f"Default cast result: {default_cast}")