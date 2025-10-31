import numpy.char as char

s = '0'
old = '00'
new = 'REPLACEMENT'

py_result = s.replace(old, new)
np_result = str(char.replace(s, old, new))

print(f"Python: '{s}'.replace('{old}', '{new}') = {repr(py_result)}")
print(f"NumPy:  char.replace('{s}', '{old}', '{new}') = {repr(np_result)}")

# Test with a few more examples to understand the pattern
test_cases = [
    ('0', '00', 'REPLACEMENT'),
    ('a', 'ab', 'XYZ'),
    ('x', 'xyz', '123456'),
    ('1', '123', 'ABCDEFG'),
]

print("\nAdditional test cases:")
for s, old, new in test_cases:
    py_result = s.replace(old, new)
    np_result = str(char.replace(s, old, new))
    print(f"s='{s}', old='{old}', new='{new}'")
    print(f"  Python: {repr(py_result)}")
    print(f"  NumPy:  {repr(np_result)}")
    print()