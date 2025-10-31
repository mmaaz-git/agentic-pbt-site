#!/usr/bin/env python3
"""Check NumPy dirichlet parameter validation in detail."""

import numpy as np
import numpy.random as npr

print("Testing parameter validation in numpy.random.dirichlet")
print("=" * 60)

rng = npr.default_rng(42)

# Test cases with different alpha values
test_cases = [
    ("All zeros", [0.0, 0.0, 0.0]),
    ("One zero", [0.0, 1.0, 1.0]),
    ("Negative value", [-1.0, 1.0, 1.0]),
    ("Mix neg and pos", [-0.5, 0.5, 1.0]),
    ("Very small positive", [1e-10, 1e-10, 1e-10]),
    ("All positive", [1.0, 2.0, 3.0]),
]

for name, alpha in test_cases:
    print(f"\nTest: {name}")
    print(f"Alpha: {alpha}")
    try:
        result = rng.dirichlet(alpha)
        print(f"Result: {result}")
        print(f"Sum: {result.sum()}")

        # Check if result satisfies Dirichlet constraints
        all_positive = np.all(result >= 0)
        sums_to_one = np.isclose(result.sum(), 1.0)
        print(f"All elements >= 0: {all_positive}")
        print(f"Sums to 1.0: {sums_to_one}")

        if not sums_to_one:
            print("⚠️  VIOLATION: Does not sum to 1.0!")

    except ValueError as e:
        print(f"ValueError raised: {e}")
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")
    print("-" * 40)

# Check what the documentation says about the constraint
print("\n" + "=" * 60)
print("Checking if negative values are properly rejected...")
print()

# According to the documentation, negative values should raise ValueError
negative_tests = [
    [-1.0, 1.0],
    [1.0, -0.001],
    [-0.5, -0.5],
]

for alpha in negative_tests:
    print(f"Testing alpha={alpha}")
    try:
        result = rng.dirichlet(alpha)
        print(f"  No error raised! Result: {result}, Sum: {result.sum()}")
    except ValueError as e:
        print(f"  ✓ ValueError raised: {e}")
    except Exception as e:
        print(f"  Unexpected error: {e}")