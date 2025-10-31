import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Create a constant series
series = pd.Series([42.0] * 20)
print(f"Input series: {series.tolist()[:5]}... (all values are 42.0)")
print(f"Series variance: {series.var()}")

# Call autocorrelation_plot
result = pd.plotting.autocorrelation_plot(series)

# Extract the autocorrelation values from the plot
lines = result.get_lines()
autocorr_line = lines[-1]  # The last line contains the autocorrelation values
ydata = autocorr_line.get_ydata()

print(f"\nAutocorrelation values (first 5): {ydata[:5]}")
print(f"All autocorrelation values are NaN: {np.all(np.isnan(ydata))}")
print(f"Number of NaN values: {np.sum(np.isnan(ydata))} out of {len(ydata)}")

plt.close('all')