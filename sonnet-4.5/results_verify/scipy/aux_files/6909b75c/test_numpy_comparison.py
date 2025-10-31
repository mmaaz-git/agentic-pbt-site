#!/usr/bin/env python3
"""Compare numpy and scipy behavior with integer p values"""

import numpy as np
from scipy import stats

print("Testing type acceptance between numpy.percentile and scipy.stats.quantile\n")

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

# Test numpy.quantile (also exists)
print("NumPy quantile tests:")
print(f"np.quantile(x, 0): {np.quantile(x, 0)}")
print(f"np.quantile(x, 1): {np.quantile(x, 1)}")
print(f"np.quantile(x, 0.5): {np.quantile(x, 0.5)}")
print()

# Check what types numpy actually uses internally
print("Type checking:")
p_int = 0
p_float = 0.0
print(f"type(0): {type(p_int)}")
print(f"type(0.0): {type(p_float)}")
print()

# Check what numpy does internally
print("NumPy's internal handling:")
result_int = np.quantile(x, p_int)
result_float = np.quantile(x, p_float)
print(f"np.quantile with int 0: {result_int}, type: {type(result_int)}")
print(f"np.quantile with float 0.0: {result_float}, type: {type(result_float)}")
print()

# Check array API compatibility
print("Array API dtype checks:")
print(f"np.isdtype(np.asarray(0).dtype, 'integral'): {np.isdtype(np.asarray(0).dtype, 'integral')}")
print(f"np.isdtype(np.asarray(0).dtype, 'real floating'): {np.isdtype(np.asarray(0).dtype, 'real floating')}")
print(f"np.isdtype(np.asarray(0.0).dtype, 'real floating'): {np.isdtype(np.asarray(0.0).dtype, 'real floating')}")