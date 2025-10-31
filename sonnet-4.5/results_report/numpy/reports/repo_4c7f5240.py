import numpy as np
import numpy.strings as nps

print("Testing numpy.strings.replace() bug:")
print("="*50)
print()

test_cases = [
    ('0', '0', '00'),
    ('a', 'a', 'aa'),
    ('hello', 'hello', 'hellohello'),
    ('hello', 'l', 'll'),
]

for s, old, new in test_cases:
    arr = np.array([s])
    py_result = s.replace(old, new)
    np_result = nps.replace(arr, old, new)[0]
    match = 'PASS' if py_result == np_result else 'FAIL'
    print(f"{match}: replace('{s}', '{old}', '{new}')")
    print(f"  Expected (Python): '{py_result}'")
    print(f"  Actual   (NumPy):  '{np_result}'")
    print()