import pandas as pd
import pandas.plotting
import matplotlib
matplotlib.use('Agg')

# Create a simple time series
series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

# Try to create a lag plot with negative lag
result = pandas.plotting.lag_plot(series, lag=-1)
print(f"Function succeeded with lag=-1")
print(f"Returned: {result}")

# Show what actually gets plotted
data = series.values
y1 = data[:-(-1)]  # This becomes data[:1]
y2 = data[-1:]      # This is data[-1:]
print(f"\nWhat gets plotted:")
print(f"y1 (x-axis) = data[:-(-1)] = data[:1] = {y1}")
print(f"y2 (y-axis) = data[-1:] = data[-1:] = {y2}")
print(f"Number of points plotted: {len(y1)} x {len(y2)} = {min(len(y1), len(y2))} points")
print(f"\nThis plots only the first element ({y1[0]}) vs the last element ({y2[0]}) - not a lag plot!")

# Show what the axis labels look like
print(f"\nAxis labels:")
print(f"x-axis: y(t)")
print(f"y-axis: y(t + {-1})")  # This would show as "y(t + -1)" which is confusing