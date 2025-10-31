import numpy as np
import numpy.char as char

arr = np.array([''])

print("find():")
print(f"  numpy.char.find([''], '\\x00') = {char.find(arr, '\x00')[0]}")
print(f"  Python ''.find('\\x00')        = {''.find(chr(0))}")

print("\nrfind():")
print(f"  numpy.char.rfind([''], '\\x00') = {char.rfind(arr, '\x00')[0]}")
print(f"  Python ''.rfind('\\x00')        = {''.rfind(chr(0))}")

print("\nstartswith():")
print(f"  numpy.char.startswith([''], '\\x00') = {char.startswith(arr, '\x00')[0]}")
print(f"  Python ''.startswith('\\x00')        = {''.startswith(chr(0))}")

print("\nendswith():")
print(f"  numpy.char.endswith([''], '\\x00') = {char.endswith(arr, '\x00')[0]}")
print(f"  Python ''.endswith('\\x00')        = {''.endswith(chr(0))}")