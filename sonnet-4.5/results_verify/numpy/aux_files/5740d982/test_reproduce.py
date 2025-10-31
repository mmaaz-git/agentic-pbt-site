import numpy.char as char
import numpy as np

print("=== Testing the specific case from bug report ===")
arr = char.array(['0'])
print(f"Created char.array(['0'])")
print(f"Array dtype: {arr.dtype}")
result = char.replace(arr, '0', '00')
print(f"After replace('0', '00'):")
print(f"Result: {result[0]!r}")
print(f"Expected: '00'")
print(f"Match: {result[0] == '00'}")
print()

print("=== Testing additional examples ===")
test_cases = [
    ('a', 'a', 'aa'),
    ('ab', 'b', 'bbb'),
    ('x', 'x', 'xyz'),
]

for s, old, new in test_cases:
    arr = char.array([s])
    result = char.replace(arr, old, new)
    expected = s.replace(old, new)
    match = result[0] == expected
    print(f"{s!r}.replace({old!r}, {new!r}): got {result[0]!r}, expected {expected!r}, match={match}")
print()

print("=== Testing with regular numpy array (dtype='U10') ===")
arr = np.array(['0'], dtype='U10')
print(f"Created np.array(['0'], dtype='U10')")
print(f"Array dtype: {arr.dtype}")
result = char.replace(arr, '0', '00')
print(f"After replace('0', '00'):")
print(f"Result: {result[0]!r}")
print(f"Expected: '00'")
print(f"Match: {result[0] == '00'}")
print()

print("=== Testing various dtypes ===")
for dtype in ['U1', 'U2', 'U5', 'U10']:
    arr = np.array(['0'], dtype=dtype)
    result = char.replace(arr, '0', '00')
    print(f"dtype={dtype}: '0' -> {result[0]!r} (expected '00', match={result[0] == '00'})")