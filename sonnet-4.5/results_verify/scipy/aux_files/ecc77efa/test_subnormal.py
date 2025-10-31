import numpy as np

# Check if the values are subnormal
values = [2.225073858507e-311, 2.003e-311, 2.225e-311]

for val in values:
    arr = np.array([val])
    is_subnormal = (val != 0) and (np.abs(val) < np.finfo(np.float64).tiny)
    print(f"Value: {val}")
    print(f"  Is non-zero: {val != 0}")
    print(f"  Is subnormal: {is_subnormal}")
    print(f"  np.finfo(float64).tiny: {np.finfo(np.float64).tiny}")
    print(f"  Comparison: {np.abs(val)} < {np.finfo(np.float64).tiny} = {np.abs(val) < np.finfo(np.float64).tiny}")
    print()