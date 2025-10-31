import numpy as np
import numpy.strings as nps

# Test the specific failing case from the bug report
s = '0'
sep = '00'

arr = np.array([s])
sep_arr = np.array([sep])

print(f"Input array dtype: {arr.dtype}")
print(f"Separator array dtype: {sep_arr.dtype}")

part1, part2, part3 = nps.partition(arr, sep_arr)
expected = s.partition(sep)

print(f"\nInput: s={s!r}, sep={sep!r}")
print(f"Python result: {expected}")
print(f"NumPy result:  ({part1[0]!r}, {part2[0]!r}, {part3[0]!r})")

try:
    assert (part1[0], part2[0], part3[0]) == expected
    print("\nAssertion passed!")
except AssertionError as e:
    print(f"\nAssertion failed!")
    print(f"Expected: {expected}")
    print(f"Got: ({part1[0]!r}, {part2[0]!r}, {part3[0]!r})")