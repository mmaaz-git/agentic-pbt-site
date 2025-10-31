import numpy as np
import scipy.signal.windows as windows

print("Testing flattop window maximum values for odd M:")
print("=" * 50)

for M in [3, 5, 7, 9, 11]:
    w = windows.flattop(M)
    max_val = np.max(w)
    print(f"flattop({M}): max = {max_val:.15f}")

print("\nTesting for even M values:")
print("=" * 50)

for M in [2, 4, 6, 8, 10]:
    w = windows.flattop(M)
    max_val = np.max(w)
    print(f"flattop({M}): max = {max_val:.15f}")

print("\nChecking if any value exceeds 1.0:")
print("=" * 50)

for M in range(1, 20):
    w = windows.flattop(M)
    max_val = np.max(w)
    if max_val > 1.0:
        print(f"M={M}: EXCEEDS 1.0 (max = {max_val:.15f})")
    else:
        print(f"M={M}: OK (max = {max_val:.15f})")