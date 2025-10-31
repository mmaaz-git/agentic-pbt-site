from pandas.io.formats.format import format_percentiles
import numpy as np

# Test various small values to find the boundary
test_values = [
    1e-45,    # Extremely small
    1e-30,
    1e-20,
    1e-15,
    1e-10,    # Suspected boundary
    1e-9,
    1e-8,
    1e-7,
    1e-6,
    1e-5,
    1e-4,
]

print("Testing various small percentiles:")
print("-" * 50)
for val in test_values:
    result = format_percentiles([val])
    print(f"{val:15.2e} -> {result[0]:10s} | Has decimal: {'.' in result[0]}")