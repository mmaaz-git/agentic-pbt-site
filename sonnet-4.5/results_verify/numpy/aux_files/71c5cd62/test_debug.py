import numpy as np
import numpy._core.records as rec

print("Testing fromrecords flow with empty list:")
recList = []

print(f"recList: {recList}")

# This is what fromrecords does internally
obj = np.array(recList, dtype=object)
print(f"obj after np.array(recList, dtype=object): {obj}")
print(f"obj.shape: {obj.shape}")

# The crash happens on this line if obj.shape[-1] is accessed for empty array
print(f"obj.shape[-1] would be: {obj.shape[-1] if len(obj.shape) > 0 else 'No dimensions'}")

# This is the problematic line in fromrecords:
if len(obj.shape) > 0:
    try:
        # This line in fromrecords causes issues
        arrlist = [np.array(obj[..., i].tolist()) for i in range(obj.shape[-1])]
        print(f"arrlist: {arrlist}")
    except Exception as e:
        print(f"Error creating arrlist: {e}")
else:
    print("obj.shape is empty")

# The function then calls fromarrays with empty arrlist
arrlist = []
print(f"\nCalling fromarrays with empty arrlist: {arrlist}")
try:
    result = rec.fromarrays(arrlist, names='x,y')
    print(f"fromarrays with empty list succeeded! Result: {result}")
except Exception as e:
    print(f"Error in fromarrays: {type(e).__name__}: {e}")