import numpy as np
import numpy.strings as nps

# Demonstrate the bug with null character
fmt_arr = np.array(['%s'], dtype='U')
val_arr = np.array(['\x00'], dtype='U')
result = nps.mod(fmt_arr, val_arr)

print(f"NumPy formatting with null character:")
print(f"  Format: {repr(fmt_arr)}")
print(f"  Value: {repr(val_arr)}")
print(f"  Result: {repr(result)}")
print(f"  Result string length: {nps.str_len(result)[0]}")
print()

# Compare with Python's behavior
python_result = '%s' % '\x00'
print(f"Python formatting with null character:")
print(f"  Format: '%s'")
print(f"  Value: '\\x00'")
print(f"  Result: {repr(python_result)}")
print(f"  Result string length: {len(python_result)}")
print()

# Test other control characters to show inconsistency
print("Testing other control characters:")
for char_name, char in [('SOH', '\x01'), ('Tab', '\t'), ('Newline', '\n')]:
    test_arr = np.array([char], dtype='U')
    test_result = nps.mod(fmt_arr, test_arr)
    print(f"  {char_name} ({repr(char)}): Result={repr(test_result)}, Length={nps.str_len(test_result)[0]}")
print()

# Bug demonstration assertions
print("Bug verification:")
print(f"  Python preserves null (length=1): {len(python_result) == 1}")
print(f"  NumPy truncates at null (length=0): {nps.str_len(result)[0] == 0}")
print(f"  NumPy result is empty string: {result[0] == ''}")
print(f"  Data loss occurred: {val_arr[0] != result[0]}")