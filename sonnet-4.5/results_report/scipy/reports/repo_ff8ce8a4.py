import numpy as np
import warnings
from scipy.spatial.distance import jensenshannon

# Test case 1: base=1.0 (should cause division by zero)
print("Test case 1: base=1.0")
p = np.array([0.5, 0.5])
q = np.array([0.3, 0.7])

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = jensenshannon(p, q, base=1.0)
    print(f"Result: {result}")
    print(f"Is infinite: {np.isinf(result)}")
    if w:
        for warning in w:
            print(f"Warning: {warning.category.__name__}: {warning.message}")
    print()

# Test case 2: base=0 (logarithm undefined)
print("Test case 2: base=0")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = jensenshannon(p, q, base=0)
    print(f"Result: {result}")
    print(f"Is nan: {np.isnan(result)}")
    if w:
        for warning in w:
            print(f"Warning: {warning.category.__name__}: {warning.message}")
    print()

# Test case 3: base=-1 (negative base, logarithm undefined)
print("Test case 3: base=-1")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = jensenshannon(p, q, base=-1)
    print(f"Result: {result}")
    print(f"Is nan: {np.isnan(result)}")
    if w:
        for warning in w:
            print(f"Warning: {warning.category.__name__}: {warning.message}")
    print()

# Test case 4: Normal valid base=2 for comparison
print("Test case 4: base=2 (valid)")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = jensenshannon(p, q, base=2)
    print(f"Result: {result}")
    print(f"Is finite: {np.isfinite(result)}")
    if w:
        for warning in w:
            print(f"Warning: {warning.category.__name__}: {warning.message}")