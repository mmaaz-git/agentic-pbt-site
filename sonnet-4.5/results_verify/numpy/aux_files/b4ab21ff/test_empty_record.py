import numpy as np
import numpy.rec

# Create empty record
dtype = np.dtype([])
arr = np.zeros(1, dtype=dtype).view(numpy.rec.recarray)
rec = arr[0]

print("Empty record created successfully")
print(f"rec type: {type(rec)}")
print(f"rec dtype: {rec.dtype}")
print(f"rec dtype.names: {rec.dtype.names}")
print(f"rec shape: {rec.shape}")
print(f"rec size: {rec.size}")

# Test other methods that might work
print("\nTesting other methods:")
try:
    print(f"str(rec): {str(rec)}")
except Exception as e:
    print(f"str(rec) failed: {e}")

try:
    print(f"repr(rec): {repr(rec)}")
except Exception as e:
    print(f"repr(rec) failed: {e}")

# Test record with fields
print("\n\nNow testing record WITH fields:")
dtype2 = np.dtype([('x', 'i4'), ('y', 'f8')])
arr2 = np.zeros(1, dtype=dtype2).view(numpy.rec.recarray)
rec2 = arr2[0]

print(f"rec2 dtype.names: {rec2.dtype.names}")
print(f"rec2.pprint():\n{rec2.pprint()}")