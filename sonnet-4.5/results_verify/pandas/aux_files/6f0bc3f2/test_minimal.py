import pandas as pd
import pandas.plotting
import matplotlib.pyplot as plt

empty_series = pd.Series([])
fig, ax = plt.subplots()

try:
    result = pandas.plotting.autocorrelation_plot(empty_series)
    print("No error occurred - function handled empty series")
except ZeroDivisionError as e:
    print(f"Error: {e}")
except ValueError as e:
    print(f"ValueError: {e}")
except Exception as e:
    print(f"Other error: {type(e).__name__}: {e}")

plt.close(fig)