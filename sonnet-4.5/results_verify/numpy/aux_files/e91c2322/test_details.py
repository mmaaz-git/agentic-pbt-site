import numpy as np
import numpy.char as char

# Test whether numpy.char actually calls Python's str methods
test_strings = ['ß', 'ῂ', 'İ']

for s in test_strings:
    print(f"\nTesting: {repr(s)}")

    # Python's behavior
    py_upper = s.upper()
    py_lower = s.lower()
    py_roundtrip = s.upper().lower()

    # NumPy's behavior
    arr = char.array([s])
    np_upper = char.upper(arr)[0]
    np_lower = char.lower(arr)[0]
    np_roundtrip = char.lower(char.upper(arr))[0]

    print(f"  Python upper: {repr(py_upper)} (len={len(py_upper)})")
    print(f"  NumPy upper:  {repr(np_upper)} (len={len(str(np_upper))})")
    print(f"  Python lower: {repr(py_lower)} (len={len(py_lower)})")
    print(f"  NumPy lower:  {repr(np_lower)} (len={len(str(np_lower))})")
    print(f"  Python roundtrip: {repr(py_roundtrip)} (len={len(py_roundtrip)})")
    print(f"  NumPy roundtrip:  {repr(np_roundtrip)} (len={len(str(np_roundtrip))})")

# Check if numpy truncates during transformation
print("\n" + "="*60)
print("Direct test of truncation:")
s = 'ß'
arr = np.array([s], dtype='U1')  # Single character Unicode string
print(f"Original array dtype: {arr.dtype}")
upper_arr = char.upper(arr)
print(f"After upper() dtype: {upper_arr.dtype}")
print(f"After upper() value: {repr(upper_arr[0])}")

# Try with larger buffer
arr2 = np.array([s], dtype='U10')  # Larger buffer
print(f"\nWith larger buffer:")
print(f"Original array dtype: {arr2.dtype}")
upper_arr2 = char.upper(arr2)
print(f"After upper() dtype: {upper_arr2.dtype}")
print(f"After upper() value: {repr(upper_arr2[0])}")