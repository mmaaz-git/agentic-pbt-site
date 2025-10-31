import numpy as np
import numpy.ctypeslib as npc

ptr_single = npc.ndpointer(flags=['C_CONTIGUOUS'])
ptr_dup = npc.ndpointer(flags=['C_CONTIGUOUS', 'C_CONTIGUOUS'])

print(f"Single flag: _flags_ = {ptr_single._flags_}")
print(f"Duplicate flags: _flags_ = {ptr_dup._flags_}")

arr = np.zeros((2, 3), dtype=np.int32, order='C')
print(f"\nArray flags.num: {arr.flags.num}")

try:
    ptr_single.from_param(arr)
    print("Single flag: PASS")
except Exception as e:
    print(f"Single flag: FAIL - {e}")

try:
    ptr_dup.from_param(arr)
    print("Duplicate flag: PASS")
except Exception as e:
    print(f"Duplicate flag: FAIL - {e}")