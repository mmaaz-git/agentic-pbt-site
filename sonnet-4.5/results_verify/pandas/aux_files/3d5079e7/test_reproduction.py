import pandas as pd
import numpy as np
import inspect

series = pd.Series([1, 2, 3])

print(f"numpy.median signature: {inspect.signature(np.median)}")
result = series.median(dtype=None)
print(f"series.median(dtype=None) = {result}")

print(f"\nnumpy.mean signature: {inspect.signature(np.mean)}")
result = series.mean(initial=None)
print(f"series.mean(initial=None) = {result}")

print(f"\nnumpy.min signature: {inspect.signature(np.min)}")
result = series.min(dtype=None)
print(f"series.min(dtype=None) = {result}")

print(f"\nnumpy.max signature: {inspect.signature(np.max)}")
result = series.max(dtype=None)
print(f"series.max(dtype=None) = {result}")

print("\n--- Testing with non-None values (should raise errors) ---")

try:
    result = series.median(dtype=float)
    print(f"series.median(dtype=float) = {result} (should have failed!)")
except Exception as e:
    print(f"series.median(dtype=float) raised: {type(e).__name__}: {e}")

try:
    result = series.mean(initial=1.0)
    print(f"series.mean(initial=1.0) = {result} (should have failed!)")
except Exception as e:
    print(f"series.mean(initial=1.0) raised: {type(e).__name__}: {e}")

try:
    result = series.min(dtype=float)
    print(f"series.min(dtype=float) = {result} (should have failed!)")
except Exception as e:
    print(f"series.min(dtype=float) raised: {type(e).__name__}: {e}")