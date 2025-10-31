import numpy as np
import numpy.strings

arr = np.array([''])
left = numpy.strings.add(numpy.strings.add(arr, '\x00'), '0')
right = numpy.strings.add(arr, '\x000')

print(f"Left:  {repr(left[0])}")
print(f"Right: {repr(right[0])}")
print(f"Equal: {np.array_equal(left, right)}")