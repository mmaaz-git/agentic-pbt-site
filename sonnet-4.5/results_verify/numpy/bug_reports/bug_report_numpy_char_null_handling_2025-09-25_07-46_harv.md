# Bug Report: numpy.char Null Character Handling

**Target**: `numpy.char.find`, `numpy.char.rfind`, `numpy.char.startswith`, `numpy.char.endswith`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

Multiple numpy.char string search functions incorrectly handle null characters (`'\x00'`), returning wrong values that don't match the documented behavior of calling Python's str methods element-wise.

## Property-Based Test

```python
import numpy as np
import numpy.char as char
from hypothesis import given, strategies as st, settings

@given(st.lists(st.text(min_size=0, max_size=50), min_size=1, max_size=10))
@settings(max_examples=500)
def test_find_matches_python_str(strings):
    arr = np.array(strings)
    numpy_results = char.find(arr, '\x00')

    for i, s in enumerate(strings):
        python_result = s.find('\x00')
        assert numpy_results[i] == python_result, f"find mismatch for {s!r}: numpy={numpy_results[i]}, python={python_result}"
```

**Failing input**: `strings=['']`

## Reproducing the Bug

```python
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
```

Output:
```
find():
  numpy.char.find([''], '\x00') = 0
  Python ''.find('\x00')        = -1

rfind():
  numpy.char.rfind([''], '\x00') = 0
  Python ''.rfind('\x00')        = -1

startswith():
  numpy.char.startswith([''], '\x00') = True
  Python ''.startswith('\x00')        = False

endswith():
  numpy.char.endswith([''], '\x00') = True
  Python ''.endswith('\x00')        = False
```

## Why This Is A Bug

The docstrings for these functions explicitly claim they call the corresponding Python str methods element-wise:
- `find()`: "For each element, return the lowest index..."
- `rfind()`: "For each element, return the highest index..."
- `startswith()`: docstring references str.startswith
- `endswith()`: docstring references str.endswith

However, when dealing with null characters, numpy.char produces different results than Python's str methods. This violates the documented API contract and could lead to incorrect results in user code that relies on these functions behaving like their Python str counterparts.

## Fix

The issue appears to be related to how numpy.char handles null-terminated strings internally. The functions likely treat null bytes as string terminators rather than regular characters, which is inconsistent with Python's str behavior.

A proper fix would require modifying the underlying C implementation to handle null bytes as regular characters throughout the string, not as terminators. The functions should use the explicit length of the numpy string arrays rather than relying on null-termination semantics.