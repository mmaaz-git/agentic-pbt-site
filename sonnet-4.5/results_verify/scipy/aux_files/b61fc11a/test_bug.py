import numpy as np
import scipy.io.matlab as sio
from io import BytesIO

# Test the specific failing case
original = np.array([0.0 + 1j * np.inf])
print(f"Original: {original}")
print(f"Original dtype: {original.dtype}")
print(f"Original real part: {original[0].real}")
print(f"Original imag part: {original[0].imag}")

f = BytesIO()
sio.savemat(f, {'x': original}, format='4')
f.seek(0)  # Reset to read
loaded = sio.loadmat(f)
result = loaded['x']

print(f"\nLoaded: {result}")
print(f"Loaded dtype: {result.dtype}")
print(f"Loaded shape: {result.shape}")
print(f"Loaded real part: {result[0,0].real}")
print(f"Loaded imag part: {result[0,0].imag}")
print(f"\nBug: real part is {result[0,0].real} (expected 0.0)")
print(f"Are they equal? {np.array_equal(original.reshape(1,1), result)}")