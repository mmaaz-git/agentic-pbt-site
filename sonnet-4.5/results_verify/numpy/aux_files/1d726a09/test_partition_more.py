import numpy as np
import numpy.strings as nps

# Test various cases
test_cases = [
    ('0', '00'),     # The original failing case
    ('a', 'ab'),     # Similar case with letters
    ('12', '123'),   # Longer input, longer separator
    ('abc', 'abcd'), # Another longer example
    ('x', 'y'),      # No match case (should work correctly)
    ('abc', 'bc'),   # Normal matching case
]

for s, sep in test_cases:
    arr = np.array([s])
    sep_arr = np.array([sep])

    part1, part2, part3 = nps.partition(arr, sep_arr)
    expected = s.partition(sep)

    print(f"Input: s={s!r} (dtype: {arr.dtype}), sep={sep!r} (dtype: {sep_arr.dtype})")
    print(f"  Python: {expected}")
    print(f"  NumPy:  ({part1[0]!r}, {part2[0]!r}, {part3[0]!r})")

    if (part1[0], part2[0], part3[0]) == expected:
        print("  ✓ MATCH")
    else:
        print("  ✗ MISMATCH")
    print()