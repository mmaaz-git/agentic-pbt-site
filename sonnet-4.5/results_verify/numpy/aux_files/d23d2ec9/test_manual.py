import numpy as np
import numpy.ma as ma

arr_correct = ma.array([1, 2, 3], mask=np.False_, shrink=True)
print(f"mask=np.False_: {ma.getmask(arr_correct)}, is nomask? {ma.getmask(arr_correct) is ma.nomask}")

arr_buggy = ma.array([1, 2, 3], mask=False, shrink=True)
print(f"mask=False: {ma.getmask(arr_buggy)}, is nomask? {ma.getmask(arr_buggy) is ma.nomask}")

# Additional debugging
print(f"\nType of mask with np.False_: {type(ma.getmask(arr_correct))}")
print(f"Type of mask with False: {type(ma.getmask(arr_buggy))}")

print(f"\nmask with np.False_ is ma.nomask: {ma.getmask(arr_correct) is ma.nomask}")
print(f"mask with False is ma.nomask: {ma.getmask(arr_buggy) is ma.nomask}")

# Check the actual values
print(f"\nnp.False_ mask value: {repr(ma.getmask(arr_correct))}")
print(f"Python False mask value: {repr(ma.getmask(arr_buggy))}")

# Verify assertions from the bug report
assert ma.getmask(arr_correct) is ma.nomask, "Expected np.False_ with shrink=True to be nomask"
assert ma.getmask(arr_buggy) is not ma.nomask, "Bug confirmed: Python False with shrink=True is not nomask"

print("\nBug confirmed: Python False does not shrink to nomask, while np.False_ does")