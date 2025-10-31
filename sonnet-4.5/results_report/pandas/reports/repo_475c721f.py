import pandas as pd
import pandas.plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import numpy as np

# Test with a constant series
constant_series = pd.Series([5.0] * 20)

print("Testing autocorrelation_plot with constant series: pd.Series([5.0] * 20)")
print("-" * 60)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", RuntimeWarning)

    try:
        result = pandas.plotting.autocorrelation_plot(constant_series)
        plt.close('all')

        runtime_warnings = [warning for warning in w if issubclass(warning.category, RuntimeWarning)]

        if runtime_warnings:
            print(f"RuntimeWarning raised: {runtime_warnings[0].message}")
            print(f"Warning category: {runtime_warnings[0].category}")
            print(f"Warning filename: {runtime_warnings[0].filename}")
            print(f"Warning lineno: {runtime_warnings[0].lineno}")
        else:
            print("No RuntimeWarning raised")

    except Exception as e:
        print(f"Exception raised: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Testing with other constant values:")
print("-" * 60)

# Test with zero constant
zero_series = pd.Series([0.0] * 20)
print("\nTesting with pd.Series([0.0] * 20):")

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", RuntimeWarning)

    try:
        result = pandas.plotting.autocorrelation_plot(zero_series)
        plt.close('all')

        runtime_warnings = [warning for warning in w if issubclass(warning.category, RuntimeWarning)]

        if runtime_warnings:
            print(f"RuntimeWarning raised: {runtime_warnings[0].message}")
        else:
            print("No RuntimeWarning raised")

    except Exception as e:
        print(f"Exception raised: {type(e).__name__}: {e}")

# Test with negative constant
negative_series = pd.Series([-10.0] * 20)
print("\nTesting with pd.Series([-10.0] * 20):")

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", RuntimeWarning)

    try:
        result = pandas.plotting.autocorrelation_plot(negative_series)
        plt.close('all')

        runtime_warnings = [warning for warning in w if issubclass(warning.category, RuntimeWarning)]

        if runtime_warnings:
            print(f"RuntimeWarning raised: {runtime_warnings[0].message}")
        else:
            print("No RuntimeWarning raised")

    except Exception as e:
        print(f"Exception raised: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Examining the actual autocorrelation values computed:")
print("-" * 60)

# Let's manually compute what happens
constant_data = np.array([5.0] * 20)
mean = np.mean(constant_data)
n = len(constant_data)
c0 = np.sum((constant_data - mean) ** 2) / n

print(f"Constant value: 5.0")
print(f"Mean: {mean}")
print(f"Variance (c0): {c0}")
print(f"c0 == 0: {c0 == 0}")

if c0 == 0:
    print("\nDivision by zero will occur in autocorrelation calculation!")
    print("The formula r(h) = sum((data[:n-h] - mean) * (data[h:] - mean)) / n / c0")
    print("When c0 = 0, this results in division by zero and NaN values")