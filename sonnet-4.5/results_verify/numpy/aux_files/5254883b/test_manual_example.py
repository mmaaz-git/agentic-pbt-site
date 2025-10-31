import numpy as np
import numpy.ctypeslib as npc

print("Testing string shape:")
ptr_str = npc.ndpointer(shape="abc")
print(f"String shape 'abc' -> _shape_ = {ptr_str._shape_}")

print("\nTesting dict shape:")
ptr_dict = npc.ndpointer(shape={"x": 1, "y": 2})
print(f"Dict shape -> _shape_ = {ptr_dict._shape_}")

print("\nTesting error message when validating array:")
arr = np.zeros((3,), dtype=np.int32)
try:
    ptr_str.from_param(arr)
except TypeError as e:
    print(f"Confusing error: {e}")

print("\nTesting set shape:")
ptr_set = npc.ndpointer(shape={"a", "b", "c"})
print(f"Set shape -> _shape_ = {ptr_set._shape_}")

print("\nTesting string '100':")
ptr_100 = npc.ndpointer(shape="100")
print(f"String shape '100' -> _shape_ = {ptr_100._shape_}")
print(f"This creates a nonsensical shape of individual characters!")