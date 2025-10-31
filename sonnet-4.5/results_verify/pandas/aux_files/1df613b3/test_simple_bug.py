import pandas as pd
import pandas.plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

constant_series = pd.Series([5.0] * 20)

print("Testing with constant series [5.0] * 20...")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", RuntimeWarning)
    result = pandas.plotting.autocorrelation_plot(constant_series)
    plt.close('all')

    runtime_warnings = [warning for warning in w if issubclass(warning.category, RuntimeWarning)]
    if runtime_warnings:
        print(f"RuntimeWarning: {runtime_warnings[0].message}")
    else:
        print("No runtime warning raised")

# Test with zero constant series
print("\nTesting with constant series [0.0] * 20...")
zero_series = pd.Series([0.0] * 20)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", RuntimeWarning)
    result = pandas.plotting.autocorrelation_plot(zero_series)
    plt.close('all')

    runtime_warnings = [warning for warning in w if issubclass(warning.category, RuntimeWarning)]
    if runtime_warnings:
        print(f"RuntimeWarning: {runtime_warnings[0].message}")
    else:
        print("No runtime warning raised")