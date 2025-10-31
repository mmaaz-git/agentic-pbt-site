import numpy as np
import numpy.ctypeslib as npc

def as_ctypes_fixed(obj):
    """Fixed version of as_ctypes that handles structured dtypes"""
    ai = obj.__array_interface__
    if ai["strides"]:
        raise TypeError("strided arrays not supported")
    if ai["version"] != 3:
        raise TypeError("only __array_interface__ version 3 supported")
    addr, readonly = ai["data"]
    if readonly:
        raise TypeError("readonly arrays unsupported")

    # Use 'descr' for structured dtypes, 'typestr' for simple dtypes
    if 'descr' in ai and len(ai['descr']) > 0 and ai['descr'][0][0]:
        # Non-empty field name means structured dtype
        ctype_scalar = npc.as_ctypes_type(ai['descr'])
    else:
        ctype_scalar = npc.as_ctypes_type(ai["typestr"])

    # This would be the rest of the implementation
    # result_type = npc._ctype_ndarray(ctype_scalar, ai["shape"])
    # result = result_type.from_address(addr)
    # result.__keep = obj
    # return result

    return ctype_scalar  # Just return the scalar type for testing

# Test with structured dtype
struct_dtype = np.dtype([('x', 'i4'), ('y', 'f8')])
arr = np.zeros(5, dtype=struct_dtype)
arr = np.ascontiguousarray(arr)
arr.flags.writeable = True

print("Testing fixed version with structured array:")
try:
    result = as_ctypes_fixed(arr)
    print(f"  Success! Got ctype: {result}")
except Exception as e:
    print(f"  Failed: {e}")

# Test with simple dtype
simple_arr = np.zeros(5, dtype='i4')
print("\nTesting fixed version with simple array:")
try:
    result = as_ctypes_fixed(simple_arr)
    print(f"  Success! Got ctype: {result}")
except Exception as e:
    print(f"  Failed: {e}")