import numpy as np

# Check the sum of coefficients
a = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]
total = sum(a)
print(f"Sum of coefficients: {total:.15f}")
print(f"Exceeds 1.0 by: {total - 1.0:.15e}")
print(f"Exceeds 1.0? {total > 1.0}")

# The proposed fix coefficients
a_fixed = [0.21557894935326313, 0.41663157875010526, 0.27726315716821054, 0.08357894674926315, 0.006947367979157896]
total_fixed = sum(a_fixed)
print(f"\nProposed fixed coefficients sum: {total_fixed:.15f}")
print(f"Equals 1.0? {total_fixed == 1.0}")

# Check how much each coefficient changed
print(f"\nChanges in coefficients:")
for i, (orig, fixed) in enumerate(zip(a, a_fixed)):
    diff = fixed - orig
    print(f"  a[{i}]: {diff:.15e}")