import numpy as np
import numpy.strings as nps

s = '\x00'
python_result = s.strip()
numpy_result = nps.strip(np.array([s]))[0]

print(f"Python str.strip(): {repr(python_result)}")
print(f"numpy.strings.strip(): {repr(numpy_result)}")

print(f"\nComparison:")
print(f"  python_result == '\\x00': {python_result == '\x00'}")
print(f"  numpy_result == '': {numpy_result == ''}")
print(f"  python_result == numpy_result: {python_result == numpy_result}")

assert python_result == '\x00'
print("✓ Python preserves null byte")

assert numpy_result == ''
print("✓ NumPy removes null byte")

print("\nTesting lstrip and rstrip as well...")

# Test lstrip
s_left = '\x00abc'
python_lstrip = s_left.lstrip()
numpy_lstrip = nps.lstrip(np.array([s_left]))[0]
print(f"\nLstrip on '\\x00abc':")
print(f"  Python: {repr(python_lstrip)}")
print(f"  NumPy:  {repr(numpy_lstrip)}")

# Test rstrip
s_right = 'abc\x00'
python_rstrip = s_right.rstrip()
numpy_rstrip = nps.rstrip(np.array([s_right]))[0]
print(f"\nRstrip on 'abc\\x00':")
print(f"  Python: {repr(python_rstrip)}")
print(f"  NumPy:  {repr(numpy_rstrip)}")