import numpy as np
import numpy.strings as nps

print("=== Understanding dtype expansion behavior ===\n")

# When does it expand vs truncate?
test_cases = [
    ('a', 'a', 'XX'),      # U1 -> should become U2
    ('ab', 'a', 'XX'),     # U2 -> should become U3
    ('abc', 'a', 'XX'),    # U3 -> should become U4
    ('abcd', 'a', 'XX'),   # U4 -> should become U5
    ('0', '0', '00'),      # The reported bug case
    ('00', '0', 'XX'),     # U2, replacing single char
    ('test', 't', 'XX'),   # Multiple occurrences
]

for input_str, old, new in test_cases:
    arr = np.array([input_str])
    result = nps.replace(arr, old, new)
    expected = input_str.replace(old, new)

    print(f"Input: '{input_str}' (dtype: {arr.dtype})")
    print(f"Replace '{old}' -> '{new}'")
    print(f"NumPy:   '{result[0]}' (dtype: {result.dtype})")
    print(f"Python:  '{expected}'")
    print(f"Match: {result[0] == expected}")
    print("-" * 40)

print("\n=== Testing with pre-allocated larger dtype ===\n")
# What if we pre-allocate with a larger dtype?
for dtype_size in [1, 5, 10]:
    arr = np.array(['a'], dtype=f'<U{dtype_size}')
    result = nps.replace(arr, 'a', 'XXXXX')
    print(f"Input dtype: <U{dtype_size}")
    print(f"Result: '{result[0]}' (dtype: {result.dtype})")
    print(f"Correct: {result[0] == 'XXXXX'}")
    print()