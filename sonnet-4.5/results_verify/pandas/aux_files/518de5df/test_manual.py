import matplotlib
matplotlib.use('Agg')

import pandas as pd
import pandas.plotting as plotting

series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

print("Test with lag=-1:")
result = plotting.lag_plot(series, lag=-1)
collections = result.collections[0]
offsets = collections.get_offsets()
print(f"Created {offsets.shape[0]} point (should have raised ValueError)")
print(f"Y-axis label: '{result.get_ylabel()}'")

print("\nTest with lag=0:")
try:
    result = plotting.lag_plot(series, lag=0)
except Exception as e:
    print(f"Raised {type(e).__name__}: {e}")