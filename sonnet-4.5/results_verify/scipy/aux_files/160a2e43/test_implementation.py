import numpy as np
from scipy.signal.windows import flattop

# Check how the flattop window is calculated for M=3, sym=True
M = 3
a = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]

# Create the window manually to understand the implementation
n = np.arange(0, M)
fac = n * 2 * np.pi / (M - 1)  # For sym=True

# Calculate the window
w = np.zeros(M)
for k in range(len(a)):
    cos_term = np.cos(k * fac)
    w += a[k] * cos_term
    if k == 0:
        print(f"k={k}: a[{k}] * cos({k}*fac) = {a[k]} * cos(0) = {a[k]} * {cos_term}")
    else:
        print(f"k={k}: a[{k}] * cos({k}*fac) = {a[k]} * {cos_term}")

print(f"\nWindow values: {w}")
print(f"Maximum value: {np.max(w):.15f}")

# At the center position (n=1 for M=3), fac = pi
print(f"\nAt center position (n=1):")
print(f"fac = {np.pi:.15f}")
for k in range(len(a)):
    cos_val = np.cos(k * np.pi)
    print(f"cos({k}*π) = {cos_val}")

# So at center: w = a[0]*1 + a[1]*(-1) + a[2]*1 + a[3]*(-1) + a[4]*1
center_val = a[0] - a[1] + a[2] - a[3] + a[4]
print(f"\nCenter value calculation:")
print(f"  {a[0]} - {a[1]} + {a[2]} - {a[3]} + {a[4]}")
print(f"  = {center_val:.15f}")

# Actually for odd M with sym=False, the issue is different
print(f"\n\nFor M=2, sym=False:")
M = 2
n = np.arange(0, M)
fac = n * 2 * np.pi / M  # For sym=False
print(f"fac values: {fac}")

# At n=0, fac=0, so all cosines = 1
w0 = sum(a[k] * np.cos(k * 0) for k in range(len(a)))
print(f"At n=0: all cos terms = 1, so w[0] = sum(a) = {w0:.15f}")

# Compare with actual scipy result
actual = flattop(2, sym=False)
print(f"Actual flattop(2, sym=False) = {actual}")
print(f"Max value = {np.max(actual):.15f}")