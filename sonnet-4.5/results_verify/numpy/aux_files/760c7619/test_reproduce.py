import numpy as np
import numpy.char

test_string = ''
sep = '\x00'

arr = np.array([test_string])

python_result = test_string.rpartition(sep)
print(f"Python rpartition: {python_result}")

try:
    numpy_result = numpy.char.rpartition(arr, sep)
    print(f"NumPy rpartition: {tuple(numpy_result[0])}")
except ValueError as e:
    print(f"NumPy rpartition: ValueError: {e}")