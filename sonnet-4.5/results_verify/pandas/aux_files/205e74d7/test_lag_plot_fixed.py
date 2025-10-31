#!/usr/bin/env python3

import pandas as pd
import numpy as np
import pandas.plotting
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

print("Testing lag_plot with invalid lag values...")

# Test 1: Basic reproduction case from bug report
print("\n1. Basic reproduction case from bug report:")
series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
print(f"Series: pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])")
print(f"Series length: {len(series)}")

lag = 10
print(f"\nCalling lag_plot with lag={lag} (which is >= series length)")
try:
    fig, ax = plt.subplots()
    result = pandas.plotting.lag_plot(series, lag=lag, ax=ax)
    print(f"✓ Function returned successfully: {type(result)}")

    # Check if any data points were actually plotted
    collections = ax.collections
    if collections:
        for i, coll in enumerate(collections):
            offsets = coll.get_offsets()
            print(f"✓ Scatter plot created with {len(offsets)} data points")
            if len(offsets) == 0:
                print("✗ BUT the plot is EMPTY - no data points visible!")
    plt.close(fig)
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

# Test 2: Show what happens internally
print("\n2. What happens internally with data slicing:")
data = series.values
print(f"Original data: {data}")
print("\nWhen lag=10 (>= len(series)=5):")
y1 = data[:-10]
y2 = data[10:]
print(f"  y1 = data[:-10] = {y1} (empty array!)")
print(f"  y2 = data[10:]  = {y2} (empty array!)")
print(f"  ax.scatter(y1={y1}, y2={y2}) creates empty plot")

print("\nFor comparison, when lag=2 (valid):")
y1 = data[:-2]
y2 = data[2:]
print(f"  y1 = data[:-2] = {y1}")
print(f"  y2 = data[2:]  = {y2}")
print(f"  ax.scatter would plot {len(y1)} points")

# Test 3: Test different lag values
print("\n3. Testing various lag values:")
test_lags = [0, 1, 4, 5, 6, 10, -1]
series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

for lag in test_lags:
    print(f"\nlag={lag}:")
    try:
        fig, ax = plt.subplots()
        result = pandas.plotting.lag_plot(series, lag=lag, ax=ax)
        collections = ax.collections
        if collections and len(collections) > 0:
            n_points = len(collections[0].get_offsets())
            if lag >= len(series):
                print(f"  ✗ lag >= len(series): Created empty plot with {n_points} points")
            elif lag < 0:
                print(f"  ? Negative lag: Created plot with {n_points} points")
            elif lag == 0:
                print(f"  ? Zero lag: Created plot with {n_points} points")
            else:
                print(f"  ✓ Valid lag: Created plot with {n_points} points")
        plt.close(fig)
    except Exception as e:
        print(f"  ! Exception: {type(e).__name__}: {e}")

# Test 4: Check with zero-lag
print("\n4. Special test for lag=0:")
try:
    fig, ax = plt.subplots()
    result = pandas.plotting.lag_plot(series, lag=0, ax=ax)
    collections = ax.collections
    if collections:
        n_points = len(collections[0].get_offsets())
        print(f"lag=0 created plot with {n_points} points")
        # When lag=0: y1=data[:-0]=data[:]=all data, y2=data[0:]=all data
        # So it plots all points against themselves
    plt.close(fig)
except Exception as e:
    print(f"lag=0 raised: {type(e).__name__}: {e}")

# Test 5: Negative lag
print("\n5. Special test for negative lag:")
try:
    fig, ax = plt.subplots()
    result = pandas.plotting.lag_plot(series, lag=-1, ax=ax)
    collections = ax.collections
    if collections:
        offsets = collections[0].get_offsets()
        n_points = len(offsets)
        print(f"lag=-1 created plot with {n_points} points")
        if n_points > 0:
            print(f"First few points: {offsets[:3]}")
    plt.close(fig)
except Exception as e:
    print(f"lag=-1 raised: {type(e).__name__}: {e}")

print("\n" + "="*60)
print("SUMMARY:")
print("="*60)
print("✗ lag_plot does NOT validate the lag parameter")
print("✗ When lag >= len(series), it creates EMPTY plots (0 data points)")
print("✗ No error or warning is raised to inform the user")
print("✗ The empty plot could be mistaken for missing data")
print("\nExpected behavior: Should raise ValueError for invalid lag values")