#!/usr/bin/env python3
"""
Minimal reproduction of the pandas.plotting.autocorrelation_plot bug
with empty Series causing ZeroDivisionError
"""

import pandas as pd
import pandas.plotting
import matplotlib.pyplot as plt

# Create an empty series
empty_series = pd.Series([])

# Create figure for plotting
fig, ax = plt.subplots()

try:
    # This should either work gracefully or raise a meaningful error
    # Instead it crashes with ZeroDivisionError
    result = pandas.plotting.autocorrelation_plot(empty_series)
    print("Success: Function returned:", result)
except ZeroDivisionError as e:
    print(f"ZeroDivisionError: {e}")
except ValueError as e:
    print(f"ValueError: {e}")
except Exception as e:
    print(f"Unexpected error ({type(e).__name__}): {e}")
finally:
    plt.close(fig)