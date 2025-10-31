import numpy.random as nr

# Test the specific failing input
print("Testing specific failing input: low=-5e-324, high=0.0")
print("="*60)

nr.seed(42)
result = nr.uniform(-5e-324, 0.0)
print(f"Result with seed 42: {result}")
print(f"Result type: {type(result)}")
print(f"Result == 0.0: {result == 0.0}")
print(f"Result >= 0.0: {result >= 0.0}")
print(f"Violates upper bound: {result >= 0.0}")
print()

# Test with multiple iterations
violations = 0
nr.seed(123)
for i in range(10000):
    result = nr.uniform(-5e-324, 0.0)
    if result >= 0.0:
        violations += 1

print(f"Violations with seed 123: {violations}/10000 ({100*violations/10000:.1f}%)")
print()

# Check the value -5e-324
print("Checking the value -5e-324:")
val = -5e-324
print(f"Value: {val}")
print(f"Is it the smallest negative float? {val == -float('inf')}: False")
print(f"Is it zero? {val == 0}: {val == 0}")
print(f"Is it less than zero? {val < 0}: {val < 0}")
print(f"Difference from 0: {0 - val}")
print()

# Additional tests with different seeds
print("Testing with additional seeds:")
seeds = [1, 2, 3, 4, 5, 10, 100, 200, 300, 400]
for seed in seeds:
    nr.seed(seed)
    violations = 0
    for i in range(1000):
        result = nr.uniform(-5e-324, 0.0)
        if result >= 0.0:
            violations += 1
    print(f"Seed {seed:3d}: {violations}/1000 violations ({100*violations/1000:.1f}%)")