import numpy as np

# Test if the issue is in array creation or in multiply
print("Testing array creation directly:")
test_strings = [
    '\x00',
    'test\x00',
    '\x00test',
    'te\x00st',
    '\x00\x00',
    'test\x00\x00'
]

for s in test_strings:
    arr = np.array([s])
    print(f"Input: {s!r} (len={len(s)}) -> Array element: {arr[0]!r} (len={len(arr[0])})")

print("\n--- Testing with dtype=object ---")
for s in test_strings:
    arr = np.array([s], dtype=object)
    print(f"Input: {s!r} (len={len(s)}) -> Array element: {arr[0]!r} (len={len(arr[0])})")

print("\n--- Testing multiply with dtype=object ---")
for s in test_strings:
    arr = np.array([s], dtype=object)
    result = np.char.multiply(arr, 1)
    print(f"Input: {s!r} * 1 -> Result: {result[0]!r} (len={len(result[0])})")