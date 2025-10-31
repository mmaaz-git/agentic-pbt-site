import numpy as np
import numpy.strings as nps

test_cases = [
    ('\x00', 3),
    ('\x00\x00', 2),
    ('a\x00', 2),
]

print("Testing numpy.strings.multiply with null characters:")
print("=" * 60)

for s, n in test_cases:
    arr = np.array([s], dtype=str)
    result = nps.multiply(arr, n)[0]
    expected = s * n

    # Print the test case
    print(f"Input string: {repr(s)}, Repetitions: {n}")
    print(f"  Python's '*' operator: {repr(expected)}")
    print(f"  numpy.strings.multiply: {repr(result)}")

    # Check if they match
    if result == expected:
        print(f"  Result: PASS")
    else:
        print(f"  Result: FAIL - Results do not match!")
        print(f"    Expected length: {len(expected)}, Got length: {len(result)}")
    print("-" * 60)