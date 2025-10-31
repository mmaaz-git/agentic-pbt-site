import numpy as np
import numpy.strings as nps

s1, s2 = 'a', 'a\x00'
arr1 = np.array([s1], dtype=str)
arr2 = np.array([s2], dtype=str)

print(f"Python strings: s1={repr(s1)}, s2={repr(s2)}")
print(f"NumPy arrays: arr1={repr(arr1)}, arr2={repr(arr2)}")
print(f"Array elements: arr1[0]={repr(arr1[0])}, arr2[0]={repr(arr2[0])}")
print()

ops = [
    ('not_equal', nps.not_equal, lambda a, b: a != b),
    ('less', nps.less, lambda a, b: a < b),
    ('greater_equal', nps.greater_equal, lambda a, b: a >= b),
]

for name, np_op, py_op in ops:
    np_result = np_op(arr1, arr2)[0]
    py_result = py_op(s1, s2)
    print(f"{name:15}: Python={py_result}, NumPy={np_result}")