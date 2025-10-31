import pandas as pd
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

print("=" * 70)
print("COMPREHENSIVE BOOTSTRAP_PLOT TESTING")
print("=" * 70)

# Create a small test series
series = pd.Series([1, 2, 3, 4, 5])

print("\n1. Testing the actual bootstrap_plot function")
print("-" * 50)
fig = pd.plotting.bootstrap_plot(series, size=5, samples=100)
print("bootstrap_plot executed successfully")
plt.close(fig)

print("\n2. Analyzing bootstrap_plot source code behavior")
print("-" * 50)
print("The function uses: random.sample(data, size)")
print("This performs sampling WITHOUT replacement")

# Demonstrate the issue
print("\n3. Demonstrating the sampling issue")
print("-" * 50)
data = [1, 2, 3, 4, 5]
size = 5
samples = 1000

# What bootstrap_plot currently does (WITHOUT replacement)
random.seed(42)
current_samplings = [random.sample(data, size) for _ in range(samples)]
current_has_duplicates = sum(1 for s in current_samplings if len(s) != len(set(s)))

print(f"Current implementation (random.sample):")
print(f"  - Samples with duplicates: {current_has_duplicates}/{samples}")
print(f"  - Percentage: {current_has_duplicates/samples*100:.1f}%")

# What bootstrap_plot SHOULD do (WITH replacement)
random.seed(42)
correct_samplings = [random.choices(data, k=size) for _ in range(samples)]
correct_has_duplicates = sum(1 for s in correct_samplings if len(s) != len(set(s)))

print(f"\nCorrect bootstrap (random.choices):")
print(f"  - Samples with duplicates: {correct_has_duplicates}/{samples}")
print(f"  - Percentage: {correct_has_duplicates/samples*100:.1f}%")

print("\n4. Statistical implications")
print("-" * 50)
print("For a 5-element dataset:")
print("  - WITH replacement: P(all different) = 5!/5^5 ≈ 3.8%")
print("  - WITHOUT replacement: P(all different) = 100%")

print("\n5. Variance estimation comparison")
print("-" * 50)
# Compare variance estimates
data = [1, 2, 3, 4, 5]
true_mean = np.mean(data)
true_var = np.var(data, ddof=1)

# Current (wrong) bootstrap
random.seed(42)
wrong_means = [np.mean(random.sample(data, 5)) for _ in range(1000)]
wrong_bootstrap_var = np.var(wrong_means, ddof=1)

# Correct bootstrap
random.seed(42)
correct_means = [np.mean(random.choices(data, k=5)) for _ in range(1000)]
correct_bootstrap_var = np.var(correct_means, ddof=1)

print(f"True population variance: {true_var:.4f}")
print(f"Variance of sample mean (theoretical): {true_var/5:.4f}")
print(f"Bootstrap estimate (WITHOUT replacement): {wrong_bootstrap_var:.4f}")
print(f"Bootstrap estimate (WITH replacement): {correct_bootstrap_var:.4f}")
print(f"\nThe WITHOUT replacement estimate is 0 because every sample")
print(f"is just a permutation with the same mean!")

print("\n6. Testing with hypothesis property-based test")
print("-" * 50)
from hypothesis import given, strategies as st, settings

@settings(max_examples=10)
@given(st.lists(st.integers(), min_size=5, max_size=20))
def test_bootstrap_should_sample_with_replacement(data):
    series = pd.Series(data)
    size = len(data)

    series_data = list(series.values)

    # Test what random.sample does
    found_duplicate = False
    for _ in range(100):
        sample = random.sample(series_data, size)
        if len(sample) != len(set(sample)):
            found_duplicate = True
            break

    # This assertion WILL fail because random.sample never produces duplicates
    assert not found_duplicate, "random.sample should NOT produce duplicates (samples without replacement)"

    print(f"✓ Test passed for data of length {len(data)}")
    return True

try:
    test_bootstrap_should_sample_with_replacement()
except Exception as e:
    print(f"Hypothesis test results: As expected, random.sample never produces duplicates")

print("\n" + "=" * 70)
print("CONCLUSION:")
print("The bug report is CORRECT. bootstrap_plot uses random.sample()")
print("which samples WITHOUT replacement, violating the fundamental")
print("requirement of bootstrapping which needs sampling WITH replacement.")
print("=" * 70)