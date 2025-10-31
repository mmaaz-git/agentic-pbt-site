import numpy as np
import numpy.ctypeslib as npc

# Create two ndpointer types - one with single flag, one with duplicate
ptr_single = npc.ndpointer(flags=['C_CONTIGUOUS'])
ptr_dup = npc.ndpointer(flags=['C_CONTIGUOUS', 'C_CONTIGUOUS'])

print(f"Single flag: _flags_ = {ptr_single._flags_}")
print(f"Duplicate flags: _flags_ = {ptr_dup._flags_}")

# Create a C-contiguous array
arr = np.zeros((2, 3), dtype=np.int32, order='C')
print(f"\nArray flags.num: {arr.flags.num}")
print(f"Array is C_CONTIGUOUS: {arr.flags['C_CONTIGUOUS']}")
print(f"Array is F_CONTIGUOUS: {arr.flags['F_CONTIGUOUS']}")

# Test with single flag (should pass)
print("\nTesting single flag pointer:")
try:
    ptr_single.from_param(arr)
    print("Single flag: PASS")
except TypeError as e:
    print(f"Single flag: FAIL - {e}")

# Test with duplicate flags (should pass but will fail)
print("\nTesting duplicate flag pointer:")
try:
    ptr_dup.from_param(arr)
    print("Duplicate flags: PASS")
except TypeError as e:
    print(f"Duplicate flags: FAIL - {e}")