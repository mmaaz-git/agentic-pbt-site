import numpy.strings as nps
import numpy as np

# Test Python's standard behavior with None
s = "hello"

# Python slice behavior
print("Python standard slice behavior:")
print(f"s[0:None] = {repr(s[0:None])}")
print(f"s[1:None] = {repr(s[1:None])}")
print(f"s[2:None] = {repr(s[2:None])}")
print(f"s[None:3] = {repr(s[None:3])}")
print(f"s[None:None] = {repr(s[None:None])}")

print("\nBuilt-in slice object behavior:")
print(f"s[slice(0, None)] = {repr(s[slice(0, None)])}")
print(f"s[slice(1, None)] = {repr(s[slice(1, None)])}")
print(f"s[slice(2, None)] = {repr(s[slice(2, None)])}")
print(f"s[slice(None, 3)] = {repr(s[slice(None, 3)])}")
print(f"s[slice(None, None)] = {repr(s[slice(None, None)])}")

print("\nnumpy.strings.slice behavior:")
arr = np.array([s])
print(f"nps.slice(arr, 0, None)[0] = {repr(nps.slice(arr, 0, None)[0])}")
print(f"nps.slice(arr, 1, None)[0] = {repr(nps.slice(arr, 1, None)[0])}")
print(f"nps.slice(arr, 2, None)[0] = {repr(nps.slice(arr, 2, None)[0])}")
print(f"nps.slice(arr, None, 3)[0] = {repr(nps.slice(arr, None, 3)[0])}")
print(f"nps.slice(arr, None, None)[0] = {repr(nps.slice(arr, None, None)[0])}")