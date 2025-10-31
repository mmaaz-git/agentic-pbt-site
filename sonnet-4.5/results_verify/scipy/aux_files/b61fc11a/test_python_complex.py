import numpy as np

# Test Python's handling of complex infinity
print("Testing Python's handling of 0 + 1j * inf:")
print()

# This is a Python/NumPy issue, not a scipy issue!
print(f"0.0 + 1j * np.inf = {0.0 + 1j * np.inf}")
print(f"complex(0.0, np.inf) = {complex(0.0, np.inf)}")

# NumPy array creation
arr1 = np.array([0.0 + 1j * np.inf])
print(f"np.array([0.0 + 1j * np.inf]) = {arr1}")

arr2 = np.array([complex(0.0, np.inf)])
print(f"np.array([complex(0.0, np.inf)]) = {arr2}")

# The real test for the bug
print("\n--- The actual bug test ---")
original = np.array([complex(0.0, np.inf)])  # This is 0+infj
print(f"Original array: {original}")
print(f"Original real part: {original[0].real}")
print(f"Original imag part: {original[0].imag}")

# Now save and load
import scipy.io.matlab as sio
from io import BytesIO

f = BytesIO()
sio.savemat(f, {'x': original}, format='4')
f.seek(0)
loaded = sio.loadmat(f)
result = loaded['x']

print(f"\nAfter save/load:")
print(f"Result array: {result}")
print(f"Result real part: {result[0,0].real}")
print(f"Result imag part: {result[0,0].imag}")

if original[0].real == 0.0 and result[0,0].real != 0.0:
    print(f"\n*** BUG CONFIRMED: Real part corrupted from {original[0].real} to {result[0,0].real} ***")