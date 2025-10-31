import numpy as np

# Check the documentation for numpy.lcm
print("=== numpy.lcm documentation ===")
print(np.lcm.__doc__)
print()

# Check if there's any mention of overflow behavior
print("=== Checking for overflow mentions ===")
if np.lcm.__doc__:
    doc = np.lcm.__doc__.lower()
    if "overflow" in doc:
        print("Found 'overflow' in documentation")
    else:
        print("No mention of 'overflow' in documentation")

    if "int64" in doc or "dtype" in doc:
        print("Found dtype/int64 mentions in documentation")
    else:
        print("No mention of int64 or dtype limitations")

# Check what dtypes are supported
print("\n=== Testing dtype behavior ===")
a = np.int32(1000000)
b = np.int32(1000000)
print(f"np.lcm with int32: {np.lcm(a, b)}, dtype: {np.lcm(a, b).dtype}")

a64 = np.int64(3036988439)
b64 = np.int64(3037012561)
print(f"np.lcm with int64: {np.lcm(a64, b64)}, dtype: {np.lcm(a64, b64).dtype}")