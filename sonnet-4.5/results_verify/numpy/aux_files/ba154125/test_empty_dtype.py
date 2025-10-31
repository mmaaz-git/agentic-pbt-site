import numpy as np
import numpy.rec

print("Testing empty dtype behavior in NumPy:")
print("-" * 50)

# Test 1: Can we create arrays with empty dtype?
print("\n1. Creating array with empty dtype using np.zeros:")
try:
    arr = np.zeros(5, dtype=[])
    print(f"Success! Shape: {arr.shape}, dtype: {arr.dtype}")
    print(f"dtype.names: {arr.dtype.names}")
    print(f"dtype.fields: {arr.dtype.fields}")
    print(f"Array: {arr}")
except Exception as e:
    print(f"Failed: {e}")

# Test 2: Can we create a structured array with no fields using np.array?
print("\n2. Creating array with empty dtype using np.array:")
try:
    arr = np.array([(), (), ()], dtype=[])
    print(f"Success! Shape: {arr.shape}, dtype: {arr.dtype}")
    print(f"Array: {arr}")
except Exception as e:
    print(f"Failed: {e}")

# Test 3: What about np.rec.array?
print("\n3. Creating record array with empty dtype using np.rec.array:")
try:
    arr = np.rec.array([(), (), ()], dtype=[])
    print(f"Success! Shape: {arr.shape}, dtype: {arr.dtype}")
    print(f"Array: {arr}")
except Exception as e:
    print(f"Failed: {e}")

# Test 4: Can we create empty structured arrays with explicit shape?
print("\n4. Creating empty structured array with explicit shape:")
try:
    arr = np.empty((3,), dtype=[])
    print(f"Success! Shape: {arr.shape}, dtype: {arr.dtype}")
    print(f"Array: {arr}")
except Exception as e:
    print(f"Failed: {e}")

# Test 5: What about fromrecords with non-empty tuples?
print("\n5. Testing fromrecords with non-empty tuples:")
try:
    arr = np.rec.fromrecords([(1, 2), (3, 4)])
    print(f"Success! Shape: {arr.shape}, dtype: {arr.dtype}")
except Exception as e:
    print(f"Failed: {e}")

# Test 6: fromrecords with explicit empty dtype
print("\n6. Testing fromrecords with explicit empty dtype:")
try:
    arr = np.rec.fromrecords([(), (), ()], dtype=[])
    print(f"Success! Shape: {arr.shape}, dtype: {arr.dtype}")
except Exception as e:
    print(f"Failed: {e}")