import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

series = pd.Series([42.0] * 20)
result = pd.plotting.autocorrelation_plot(series)

lines = result.get_lines()
autocorr_line = lines[-1]
ydata = autocorr_line.get_ydata()
print(f"Autocorrelation values: {ydata[:10]}")  # Just show first 10 values
print(f"All NaN: {np.all(np.isnan(ydata))}")