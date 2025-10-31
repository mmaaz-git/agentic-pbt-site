import numpy as np
import scipy.sparse as sp

print("Testing numpy.eye behavior:")
np_result = np.eye(1, k=2)
print(f"numpy.eye(1, k=2) = {np_result}")

print("\nTesting scipy.sparse.eye_array behavior:")
try:
    sp_result = sp.eye_array(1, k=2)
    print(f"scipy.sparse.eye_array(1, k=2) = {sp_result.toarray()}")
except ValueError as e:
    print(f"scipy.sparse.eye_array(1, k=2) raised ValueError: {e}")

print("\nTesting additional cases mentioned in bug report:")

# Test case: eye_array(2, k=3)
print("\n1. numpy.eye(2, k=3):")
np_result = np.eye(2, k=3)
print(f"   Result: {np_result}")

print("   scipy.sparse.eye_array(2, k=3):")
try:
    sp_result = sp.eye_array(2, k=3)
    print(f"   Result: {sp_result.toarray()}")
except ValueError as e:
    print(f"   Raised ValueError: {e}")

# Test case: eye_array(3, k=4)
print("\n2. numpy.eye(3, k=4):")
np_result = np.eye(3, k=4)
print(f"   Result: {np_result}")

print("   scipy.sparse.eye_array(3, k=4):")
try:
    sp_result = sp.eye_array(3, k=4)
    print(f"   Result: {sp_result.toarray()}")
except ValueError as e:
    print(f"   Raised ValueError: {e}")

# Test negative k values
print("\n3. numpy.eye(2, k=-3):")
np_result = np.eye(2, k=-3)
print(f"   Result: {np_result}")

print("   scipy.sparse.eye_array(2, k=-3):")
try:
    sp_result = sp.eye_array(2, k=-3)
    print(f"   Result: {sp_result.toarray()}")
except ValueError as e:
    print(f"   Raised ValueError: {e}")

# Test rectangular matrix
print("\n4. numpy.eye(1, 3, k=-2):")
np_result = np.eye(1, 3, k=-2)
print(f"   Result: {np_result}")

print("   scipy.sparse.eye_array(1, 3, k=-2):")
try:
    sp_result = sp.eye_array(1, 3, k=-2)
    print(f"   Result: {sp_result.toarray()}")
except ValueError as e:
    print(f"   Raised ValueError: {e}")