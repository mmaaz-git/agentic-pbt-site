import numpy as np
import numpy.strings as nps

print("=" * 60)
print("Testing numpy.strings.mod with null character")
print("=" * 60)

# Demonstrate the bug
fmt_arr = np.array(['%s'], dtype='U')
val_arr = np.array(['\x00'], dtype='U')
result = nps.mod(fmt_arr, val_arr)

print(f"Input format array: {repr(fmt_arr)}")
print(f"Input value array: {repr(val_arr)}")
print(f"Result: {repr(result)}")
print(f"Result[0]: {repr(result[0])}")
print(f"Result length (using nps.str_len): {nps.str_len(result)[0]}")

print("\n" + "=" * 60)
print("Compare with Python's behavior")
print("=" * 60)

python_result = '%s' % '\x00'
print(f"Python result: {repr(python_result)}")
print(f"Python result length: {len(python_result)}")

print("\n" + "=" * 60)
print("Bug demonstration assertions")
print("=" * 60)

# Bug demonstration
try:
    assert len(python_result) == 1
    print("✓ Python correctly handles null (length = 1)")
except AssertionError:
    print("✗ Python assertion failed")

try:
    assert nps.str_len(result)[0] == 0
    print("✓ NumPy truncates at null - result length is 0 (BUG)")
except AssertionError:
    print("✗ NumPy does not truncate at null")

try:
    assert result[0] == ''
    print("✓ Result is empty string instead of '\\x00' (BUG)")
except AssertionError:
    print("✗ Result is not empty string")

print("\n" + "=" * 60)
print("Testing with other control characters for comparison")
print("=" * 60)

# Test other control characters
for char, name in [('\x01', 'SOH'), ('\t', 'Tab'), ('\n', 'Newline')]:
    fmt_arr = np.array(['%s'], dtype='U')
    val_arr = np.array([char], dtype='U')
    result = nps.mod(fmt_arr, val_arr)
    print(f"{name} (\\x{ord(char):02x}): Input={repr(char)}, Result={repr(result[0])}, Length={nps.str_len(result)[0]}")

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("The null character \\x00 is treated as a string terminator by numpy.strings.mod")
print("This differs from Python's % operator and from NumPy's handling of other control chars")