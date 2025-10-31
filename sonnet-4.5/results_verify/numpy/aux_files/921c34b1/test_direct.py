from pandas.core.window.common import zsqrt

# Test with Python float
try:
    result = zsqrt(5.0)
    print(f"zsqrt(5.0) = {result}")
except Exception as e:
    print(f"Error with Python float: {type(e).__name__}: {e}")

# Test with negative Python float
try:
    result = zsqrt(-5.0)
    print(f"zsqrt(-5.0) = {result}")
except Exception as e:
    print(f"Error with negative Python float: {type(e).__name__}: {e}")

# Test with numpy scalar
import numpy as np
try:
    result = zsqrt(np.float64(5.0))
    print(f"zsqrt(np.float64(5.0)) = {result}")
except Exception as e:
    print(f"Error with numpy scalar: {type(e).__name__}: {e}")

# Test with numpy array
try:
    result = zsqrt(np.array([1.0, 4.0, -9.0]))
    print(f"zsqrt(np.array([1.0, 4.0, -9.0])) = {result}")
except Exception as e:
    print(f"Error with numpy array: {type(e).__name__}: {e}")

# Test with pandas Series
import pandas as pd
try:
    result = zsqrt(pd.Series([1.0, 4.0, -9.0]))
    print(f"zsqrt(pd.Series([1.0, 4.0, -9.0])) = {result.tolist()}")
except Exception as e:
    print(f"Error with pandas Series: {type(e).__name__}: {e}")