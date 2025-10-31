import numpy.char as char

s = '0'
old = '00'
new = 'REPLACEMENT'

py_result = s.replace(old, new)
np_result = str(char.replace(s, old, new))

print(f"Python: '{s}'.replace('{old}', '{new}') = {repr(py_result)}")
print(f"NumPy:  char.replace('{s}', '{old}', '{new}') = {repr(np_result)}")