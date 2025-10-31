import pandas as pd
import random
import numpy as np

print("=" * 60)
print("Testing bootstrap sampling behavior")
print("=" * 60)

# Test 1: Simple reproduction of the bug
series = pd.Series([1, 2, 3])
random.seed(42)

data = list(series.values)
size = 3
samples = 1000

samplings = [random.sample(data, size) for _ in range(samples)]

has_duplicates = any(len(s) != len(set(s)) for s in samplings)

print("\nTest 1: Checking if random.sample produces duplicates")
print(f"Any sample has duplicates: {has_duplicates}")
print(f"Expected for bootstrap (WITH replacement): True")
print(f"Actual result: {has_duplicates}")

# Test 2: Verify that random.sample doesn't allow duplicates
print("\n" + "=" * 60)
print("Test 2: Demonstrating random.sample behavior")
data = [1, 2, 3, 4, 5]
try:
    # Try to sample more than available
    sample = random.sample(data, 6)
except ValueError as e:
    print(f"random.sample(data, 6) raises: {e}")
    print("This confirms random.sample samples WITHOUT replacement")

# Test 3: Show what random.choices does (correct bootstrap)
print("\n" + "=" * 60)
print("Test 3: Demonstrating random.choices behavior (correct bootstrap)")
random.seed(42)
data = [1, 2, 3]
samples_choices = [random.choices(data, k=3) for _ in range(10)]

print("First 10 samples using random.choices (WITH replacement):")
for i, sample in enumerate(samples_choices[:10]):
    duplicates = len(sample) != len(set(sample))
    print(f"  Sample {i+1}: {sample} - Has duplicates: {duplicates}")

# Count how many have duplicates
has_dups_choices = sum(1 for s in samples_choices if len(s) != len(set(s)))
print(f"\nSamples with duplicates: {has_dups_choices}/10")

# Test 4: Statistical implications
print("\n" + "=" * 60)
print("Test 4: Statistical implications")
print("\nFor a 3-element dataset sampled WITH replacement:")
print("  Probability of all different = (3/3) * (2/3) * (1/3) = 2/9 ≈ 22%")
print("  Probability of duplicates = 1 - 2/9 = 7/9 ≈ 78%")

print("\nFor a 3-element dataset sampled WITHOUT replacement:")
print("  Probability of all different = 100% (by definition)")
print("  Probability of duplicates = 0% (impossible)")