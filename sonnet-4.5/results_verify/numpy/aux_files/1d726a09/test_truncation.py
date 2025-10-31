import numpy as np
import numpy.strings as nps

# Test if the separator is being truncated
s = '0'
sep = '00'

arr = np.array([s])  # dtype will be <U1
sep_arr = np.array([sep])  # dtype will be <U2

print(f"Original separator: {sep!r} (dtype: {sep_arr.dtype})")

# What happens if we cast the separator to the array's dtype?
sep_truncated = sep_arr.astype(arr.dtype)
print(f"Separator cast to {arr.dtype}: {sep_truncated[0]!r}")

# Now let's partition with the truncated separator directly
part1, part2, part3 = nps.partition(arr, sep_truncated)
print(f"\nPartitioning '0' with truncated separator '0':")
print(f"  Result: ({part1[0]!r}, {part2[0]!r}, {part3[0]!r})")

# Compare with normal Python partition
print(f"\nPython partition('0', '0'): {s.partition('0')}")
print(f"Python partition('0', '00'): {s.partition('00')}")