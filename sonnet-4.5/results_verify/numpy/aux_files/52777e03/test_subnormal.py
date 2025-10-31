import numpy as np
import sys

# Test subnormal float properties
subnormal = 2.2250738585e-313
print(f"Subnormal value: {subnormal}")
print(f"Is it a valid float? {isinstance(subnormal, float)}")
print(f"Smallest normal float: {sys.float_info.min}")
print(f"Is subnormal < smallest normal? {subnormal < sys.float_info.min}")

# Create numpy array with subnormal
a = np.array([subnormal])
print(f"\nNumpy array with subnormal: {a}")
print(f"Array dtype: {a.dtype}")
print(f"Is finite? {np.isfinite(a[0])}")
print(f"Is NaN? {np.isnan(a[0])}")
print(f"Is Inf? {np.isinf(a[0])}")

# Test what happens with reciprocal
print(f"\n1/subnormal = {1/subnormal}")
print(f"Is 1/subnormal infinite? {np.isinf(1/subnormal)}")

# Check numpy's handling of subnormal in basic operations
print(f"\nNumpy divide: {np.divide(1.0, subnormal)}")
print(f"Contains inf? {np.isinf(np.divide(1.0, subnormal))}")