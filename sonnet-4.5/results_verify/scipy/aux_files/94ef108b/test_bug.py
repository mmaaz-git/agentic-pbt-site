import numpy as np
from scipy.differentiate import derivative

def f(x):
    return x**2

x = 2.0
true_derivative = 4.0

print("=== Reproducing the reported bug ===\n")

print("Test 1: initial_step = 0")
try:
    result = derivative(f, x, initial_step=0.0)
    print(f"  Result: df={result.df}, status={result.status}, success={result.success}")
    print(f"  Is df NaN? {np.isnan(result.df)}")
    print(f"  Expected: ValueError with message about invalid initial_step")
    print(f"  Actual: Silently returns result with status={result.status}")
except ValueError as e:
    print(f"  ValueError raised: {e}")

print("\nTest 2: initial_step = -0.5")
try:
    result = derivative(f, x, initial_step=-0.5)
    print(f"  Result: df={result.df}, status={result.status}, success={result.success}")
    print(f"  Is df NaN? {np.isnan(result.df)}")
    print(f"  Expected: ValueError with message about invalid initial_step")
    print(f"  Actual: Silently returns result with status={result.status}")
except ValueError as e:
    print(f"  ValueError raised: {e}")

print("\nTest 3: initial_step = 0.5 (valid, control)")
try:
    result = derivative(f, x, initial_step=0.5)
    print(f"  Result: df={result.df}, error from true={abs(result.df - true_derivative)}")
    print(f"  Status: {result.status}, success={result.success}")
    print(f"  Works correctly as expected")
except Exception as e:
    print(f"  Unexpected error: {e}")

print("\nTest 4: initial_step = -10.0")
try:
    result = derivative(f, x, initial_step=-10.0)
    print(f"  Result: df={result.df}, status={result.status}, success={result.success}")
    print(f"  Is df NaN? {np.isnan(result.df)}")
except ValueError as e:
    print(f"  ValueError raised: {e}")

print("\n=== Running the Hypothesis test ===\n")

from hypothesis import given, strategies as st

def test_initial_step_validation(initial_step):
    """initial_step must be positive - function should raise ValueError for invalid values."""
    def f(x):
        return x**2

    try:
        result = derivative(f, 1.0, initial_step=initial_step)
        # If we get here without exception, check if result is valid
        assert not np.isnan(result.df), \
            f"Function returned NaN for initial_step={initial_step} instead of raising error"
        assert False, \
            f"Function accepted invalid initial_step={initial_step} without raising ValueError"
    except ValueError as e:
        # This is the expected behavior
        assert "initial_step" in str(e).lower() or "step" in str(e).lower(), \
            f"ValueError raised but message doesn't mention step parameter: {e}"

# Run a few examples from hypothesis
print("Testing with hypothesis-like examples:")
for step in [0.0, -0.1, -1.0, -5.0]:
    print(f"\nTesting initial_step={step}")
    try:
        test_initial_step_validation(step)
        print(f"  Test passed (ValueError was raised as expected)")
    except AssertionError as e:
        print(f"  Test failed: {e}")