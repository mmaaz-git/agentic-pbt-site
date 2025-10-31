import numpy as np

start = np.array([1.0, 0.0, 0.0])
t_scalar = 0.5
t_array = np.array([0.5])

# Convert scalar to array (as done in geometric_slerp)
t_scalar_converted = np.asarray(t_scalar)
print(f"Scalar t converted: {t_scalar_converted}, shape: {t_scalar_converted.shape}, ndim: {t_scalar_converted.ndim}, size: {t_scalar_converted.size}")

# What np.linspace returns when start == end
result = np.linspace(start, start, t_scalar_converted.size)
print(f"\nnp.linspace(start, start, t.size) with scalar t:")
print(f"Result: {result}")
print(f"Shape: {result.shape}")

# What's happening in the normal path (lines 231-234)
print(f"\nNormal path when start != end with scalar t:")
print(f"t.ndim == 0: {t_scalar_converted.ndim == 0}")
print("This path calls _geometric_slerp with np.atleast_1d(t) and then .ravel()")

# Simulating what should happen
t_1d = np.atleast_1d(t_scalar_converted)
print(f"np.atleast_1d(t): {t_1d}, shape: {t_1d.shape}")

# What the fix should return
print(f"\nWhat the fix should return when start == end and t is scalar:")
print(f"start.copy(): {start.copy()}, shape: {start.copy().shape}")