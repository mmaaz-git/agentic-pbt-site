import numpy as np
import numpy.strings as nps

s = '0'
sep = '00'

arr = np.array([s])
sep_arr = np.array([sep])

part1, part2, part3 = nps.partition(arr, sep_arr)
expected = s.partition(sep)

print(f"Input: s={s!r}, sep={sep!r}")
print(f"Python result: {expected}")
print(f"NumPy result:  ({part1[0]!r}, {part2[0]!r}, {part3[0]!r})")
print()
print(f"Expected: {expected}")
print(f"Actual:   ({part1[0]!r}, {part2[0]!r}, {part3[0]!r})")
print()

# Check if they match
if (part1[0], part2[0], part3[0]) == expected:
    print("✓ Results match")
else:
    print("✗ Results DO NOT match - BUG CONFIRMED")
    print(f"  Expected first element: {expected[0]!r}")
    print(f"  Got first element:      {part1[0]!r}")

# This will raise AssertionError
assert (part1[0], part2[0], part3[0]) == expected