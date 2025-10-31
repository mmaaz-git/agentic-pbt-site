import numpy as np
from pandas.compat.numpy.function import process_skipna

# Test with np.bool_(True)
np_true = np.bool_(True)
result_skipna, result_args = process_skipna(np_true, ())

print(f"Input: np.bool_(True)")
print(f"Return type: {type(result_skipna)}")
print(f"Return value: {result_skipna}")
print(f"Is Python bool: {type(result_skipna) == bool}")
print(f"Is np.bool_: {isinstance(result_skipna, np.bool_)}")
print()

# Test with np.bool_(False)
np_false = np.bool_(False)
result_skipna2, result_args2 = process_skipna(np_false, ())

print(f"Input: np.bool_(False)")
print(f"Return type: {type(result_skipna2)}")
print(f"Return value: {result_skipna2}")
print(f"Is Python bool: {type(result_skipna2) == bool}")
print(f"Is np.bool_: {isinstance(result_skipna2, np.bool_)}")
print()

# Test with regular Python bool
py_bool = True
result_skipna3, result_args3 = process_skipna(py_bool, ())

print(f"Input: Python True")
print(f"Return type: {type(result_skipna3)}")
print(f"Return value: {result_skipna3}")
print(f"Is Python bool: {type(result_skipna3) == bool}")
print(f"Is np.bool_: {isinstance(result_skipna3, np.bool_)}")
print()

# Run the assertion from the bug report
try:
    assert isinstance(result_skipna, bool) and not isinstance(result_skipna, np.bool_)
    print("Assertion passed")
except AssertionError:
    print("Assertion failed: result_skipna is not a Python bool but rather np.bool_")