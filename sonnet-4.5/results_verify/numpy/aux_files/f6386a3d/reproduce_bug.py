import numpy as np
import numpy.strings

arr = np.array([''])
left = numpy.strings.add(numpy.strings.add(arr, '\x00'), '0')
right = numpy.strings.add(arr, '\x000')

print(f"Left:  {repr(left[0])}")
print(f"Right: {repr(right[0])}")

# Let's also check intermediate results
intermediate = numpy.strings.add(arr, '\x00')
print(f"\nIntermediate result (add([''], '\\x00')): {repr(intermediate[0])}")
print(f"Length of intermediate result: {len(intermediate[0])}")

# After adding '0' to the intermediate
final_left = numpy.strings.add(intermediate, '0')
print(f"\nFinal left (add(intermediate, '0')): {repr(final_left[0])}")

# Direct concatenation
concat_string = '\x00' + '0'
print(f"\nDirect string concat ('\\x00' + '0'): {repr(concat_string)}")
direct_add = numpy.strings.add(arr, concat_string)
print(f"Direct add result (add([''], '\\x000')): {repr(direct_add[0])}")

print(f"\nAre results equal? {np.array_equal(left, right)}")