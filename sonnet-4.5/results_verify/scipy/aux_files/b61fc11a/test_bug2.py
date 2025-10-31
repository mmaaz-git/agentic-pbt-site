import numpy as np
import scipy.io.matlab as sio
from io import BytesIO

# Test creating the complex number different ways
print("Testing different ways to create 0+infj:")
print()

# Method 1: Direct complex construction
val1 = complex(0.0, np.inf)
print(f"complex(0.0, np.inf) = {val1}")

# Method 2: Using numpy complex
val2 = np.complex128(0.0 + 1j * np.inf)
print(f"np.complex128(0.0 + 1j * np.inf) = {val2}")

# Method 3: The problematic computation 0.0 + 1j * np.inf
val3 = 0.0 + 1j * np.inf
print(f"0.0 + 1j * np.inf = {val3}")

# Method 4: Create array directly
arr = np.array([complex(0.0, np.inf)])
print(f"np.array([complex(0.0, np.inf)]) = {arr}")

print()
print("Now testing the save/load round trip:")
print()

# Use the cleanest way to create the complex number
original = np.array([complex(0.0, np.inf)])
print(f"Original: {original}")
print(f"Original real part: {original[0].real}")
print(f"Original imag part: {original[0].imag}")

f = BytesIO()
sio.savemat(f, {'x': original}, format='4')
f.seek(0)
loaded = sio.loadmat(f)
result = loaded['x']

print(f"\nLoaded: {result}")
print(f"Loaded real part: {result[0,0].real}")
print(f"Loaded imag part: {result[0,0].imag}")

if result[0,0].real != 0.0:
    print(f"\nBUG CONFIRMED: real part changed from {original[0].real} to {result[0,0].real}")