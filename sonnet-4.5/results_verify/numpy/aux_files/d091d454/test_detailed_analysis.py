import numpy.random as nr
import numpy as np
import sys

print("Detailed analysis of numpy.random.uniform with tiny ranges")
print("="*70)

# Check what exactly -5e-324 is
print("\n1. Understanding -5e-324:")
val = -5e-324
print(f"   Value: {val}")
print(f"   Representation: {val!r}")
print(f"   Hex representation: {val.hex()}")
print(f"   Is subnormal: {abs(val) < sys.float_info.min}")
print(f"   Smallest normal float: {sys.float_info.min}")
print(f"   Smallest subnormal float: {sys.float_info.min * sys.float_info.epsilon}")
print(f"   Machine epsilon: {sys.float_info.epsilon}")

# Check if returned values are actually exactly 0.0
print("\n2. Checking if violations are exactly 0.0:")
nr.seed(123)
exact_zeros = 0
violations = []
for i in range(100):
    result = nr.uniform(-5e-324, 0.0)
    if result >= 0.0:
        violations.append(result)
        if result == 0.0:
            exact_zeros += 1

print(f"   Total violations: {len(violations)}")
print(f"   Exact zeros: {exact_zeros}")
print(f"   First 10 violating values: {violations[:10]}")

# Test with different tiny ranges
print("\n3. Testing with different tiny ranges:")
test_ranges = [
    (-1e-323, 0.0),
    (-1e-324, 0.0),
    (-5e-324, 0.0),
    (-1e-325, 0.0),  # Even smaller
    (-sys.float_info.min, 0.0),  # Smallest normal
    (-sys.float_info.min * sys.float_info.epsilon, 0.0),  # Smallest subnormal
]

for low, high in test_ranges:
    nr.seed(123)
    violations = 0
    for i in range(1000):
        result = nr.uniform(low, high)
        if result >= high:
            violations += 1
    print(f"   Range [{low:.3e}, {high}): {violations}/1000 violations ({100*violations/1000:.1f}%)")

# Check floating point arithmetic
print("\n4. Checking floating point arithmetic:")
low = -5e-324
high = 0.0
diff = high - low
print(f"   low = {low}")
print(f"   high = {high}")
print(f"   high - low = {diff}")
print(f"   Is (high - low) == 5e-324? {diff == 5e-324}")
print(f"   low + (high - low) * 1.0 = {low + diff * 1.0}")
print(f"   Does low + (high - low) * 1.0 == high? {low + diff * 1.0 == high}")

# Simulate the likely implementation
print("\n5. Simulating likely implementation:")
print("   Formula: low + (high - low) * random_unit")
random_units = [0.0, 0.5, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 0.9999999, 0.99999999, 1.0 - sys.float_info.epsilon]
for ru in random_units:
    result = low + (high - low) * ru
    print(f"   random_unit={ru:.10f}: result={result:.3e}, >= high: {result >= high}")

# Check the new Generator API
print("\n6. Testing with new Generator API (recommended by numpy docs):")
rng = np.random.default_rng(123)
violations_new = 0
for i in range(1000):
    result = rng.uniform(-5e-324, 0.0)
    if result >= 0.0:
        violations_new += 1
print(f"   New API violations: {violations_new}/1000 ({100*violations_new/1000:.1f}%)")

# Test nextafter solution
print("\n7. Testing potential fix with nextafter:")
test_val = 0.0
next_down = np.nextafter(0.0, -1.0)
print(f"   nextafter(0.0, -1.0) = {next_down}")
print(f"   Is it less than 0.0? {next_down < 0.0}")
print(f"   Difference from 0.0: {0.0 - next_down}")