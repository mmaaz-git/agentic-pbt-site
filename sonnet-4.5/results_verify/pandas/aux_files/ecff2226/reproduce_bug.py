import pandas as pd
import pandas.plotting
import matplotlib
matplotlib.use('Agg')

series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

result = pandas.plotting.lag_plot(series, lag=-1)
print(f"Function succeeded with lag=-1")
print(f"Returned: {result}")

data = series.values
y1 = data[:-(-1)]
y2 = data[-1:]
print(f"\nWhat gets plotted:")
print(f"y1 = data[:1] = {y1}")
print(f"y2 = data[-1:] = {y2}")
print(f"This plots only the first element vs the last element - not a lag plot!")

# Let's also verify the slicing behavior
print(f"\nVerifying slicing behavior:")
print(f"data = {data}")
print(f"data[:-(-1)] = data[:1] = {data[:1]}")
print(f"data[-1:] = {data[-1:]}")

# Test with different negative lag values
for lag in [-1, -2, -3, -4]:
    y1 = data[:-lag]
    y2 = data[-lag:]
    print(f"\nlag={lag}: y1=data[:{-lag}]={y1[:5]}..., y2=data[{-lag}:]={y2[:5]}...")
    print(f"  y1 length: {len(y1)}, y2 length: {len(y2)}")
    print(f"  Scatter points: {min(len(y1), len(y2))}")