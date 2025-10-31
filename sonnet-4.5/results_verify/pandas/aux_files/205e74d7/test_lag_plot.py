#!/usr/bin/env python3

import pandas as pd
import numpy as np
import pandas.plotting
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

print("Testing lag_plot with invalid lag values...")

# Test 1: Basic reproduction case
print("\n1. Basic reproduction case:")
series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
print(f"Series length: {len(series)}")
print(f"Series values: {series.values}")

# Test with lag >= len(series)
lag = 10
print(f"\nTrying lag_plot with lag={lag} (which is >= series length)")
try:
    fig, ax = plt.subplots()
    result = pandas.plotting.lag_plot(series, lag=lag, ax=ax)
    print(f"Result type: {type(result)}")
    print(f"Plot created successfully (no error raised)")

    # Check if any data points were actually plotted
    collections = ax.collections
    if collections:
        for i, coll in enumerate(collections):
            offsets = coll.get_offsets()
            print(f"Collection {i}: {len(offsets)} data points plotted")
            if len(offsets) == 0:
                print("  -> Empty scatter plot!")
    else:
        print("No collections in the plot")
    plt.close(fig)
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

# Test 2: Verify what data arrays are created internally
print("\n2. Checking internal data slicing:")
data = series.values
print(f"Original data: {data}")
for test_lag in [1, 3, 5, 10]:
    y1 = data[:-test_lag] if test_lag > 0 else data
    y2 = data[test_lag:]
    print(f"lag={test_lag}: y1={y1}, y2={y2}, len(y1)={len(y1)}, len(y2)={len(y2)}")

# Test 3: Test edge cases
print("\n3. Testing edge cases:")
edge_cases = [
    (5, 4),  # lag < len(series) - should work
    (5, 5),  # lag == len(series) - empty plot
    (5, 10), # lag > len(series) - empty plot
    (3, 3),  # lag == len(series) - empty plot
    (10, 1), # normal case
]

for series_len, lag in edge_cases:
    series = pd.Series(np.random.randn(series_len))
    print(f"\nSeries length: {series_len}, lag: {lag}")
    try:
        fig, ax = plt.subplots()
        result = pandas.plotting.lag_plot(series, lag=lag, ax=ax)

        # Check if any data was plotted
        collections = ax.collections
        if collections and len(collections) > 0:
            n_points = len(collections[0].get_offsets())
            print(f"  -> Success: {n_points} points plotted")
            if n_points == 0:
                print(f"     WARNING: Empty plot created!")
        else:
            print(f"  -> Success but no scatter collections")
        plt.close(fig)
    except Exception as e:
        print(f"  -> Exception: {type(e).__name__}: {e}")

# Test 4: Run the hypothesis test
print("\n4. Running property-based test:")
from hypothesis import given, strategies as st, settings

@given(
    series_len=st.integers(min_value=2, max_value=100),
    lag=st.integers(min_value=1, max_value=200),
)
@settings(max_examples=20, deadline=None)
def test_lag_plot_should_validate_lag_parameter(series_len, lag):
    series = pd.Series(np.random.randn(series_len))

    if lag >= series_len:
        # This should arguably raise an error
        fig, ax = plt.subplots()
        result = pandas.plotting.lag_plot(series, lag=lag, ax=ax)
        collections = ax.collections
        if collections:
            n_points = len(collections[0].get_offsets())
            assert n_points == 0, f"Expected empty plot but got {n_points} points"
        plt.close(fig)
        return f"lag={lag}, len={series_len}: empty plot created"
    else:
        # Valid lag values should work correctly
        fig, ax = plt.subplots()
        result = pandas.plotting.lag_plot(series, lag=lag, ax=ax)
        collections = ax.collections
        if collections:
            n_points = len(collections[0].get_offsets())
            expected_points = series_len - lag
            assert n_points == expected_points, f"Expected {expected_points} points, got {n_points}"
        plt.close(fig)
        return f"lag={lag}, len={series_len}: {series_len - lag} points plotted"

# Run a few examples
print("Testing with various series lengths and lag values:")
test_cases = [
    (5, 3),   # Valid
    (5, 5),   # Edge case: lag == len
    (5, 10),  # Invalid: lag > len
    (10, 1),  # Valid
    (10, 15), # Invalid: lag > len
]

for series_len, lag in test_cases:
    result = test_lag_plot_should_validate_lag_parameter(series_len, lag)
    print(f"  {result}")

print("\nConclusion:")
print("lag_plot does NOT validate the lag parameter.")
print("When lag >= len(series), it creates empty plots without raising any error.")
print("This results in meaningless visualizations with no data points.")