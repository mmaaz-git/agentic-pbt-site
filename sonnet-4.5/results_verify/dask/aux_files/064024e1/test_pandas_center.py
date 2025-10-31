import pandas as pd
import numpy as np

# Test pandas with time-based windows and center=True
df = pd.DataFrame({
    'time': pd.date_range('2020-01-01', periods=10, freq='1h'),
    'value': range(10)
})
df = df.set_index('time')

print("Original DataFrame:")
print(df)
print("\n")

# Test with center=False
print("Rolling with window='2h', center=False:")
result_no_center = df.rolling(window='2h', center=False).mean()
print(result_no_center)
print("\n")

# Test with center=True
print("Rolling with window='2h', center=True:")
result_with_center = df.rolling(window='2h', center=True).mean()
print(result_with_center)
print("\n")

print("Difference between center=False and center=True:")
print("Are they different?", not result_no_center.equals(result_with_center))