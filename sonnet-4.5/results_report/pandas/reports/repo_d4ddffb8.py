import pandas as pd
import pandas.api.typing as pat

# Create a NaT instance using NaTType()
nat_from_call = pat.NaTType()

# Check if it's the same as pd.NaT singleton
print(f"nat_from_call is pd.NaT: {nat_from_call is pd.NaT}")

# Check if pd.isna recognizes it
print(f"pd.isna(nat_from_call): {pd.isna(nat_from_call)}")

# Check repr
print(f"repr(nat_from_call): {repr(nat_from_call)}")
print(f"repr(pd.NaT): {repr(pd.NaT)}")

# Check equality
print(f"nat_from_call == pd.NaT: {nat_from_call == pd.NaT}")

# Create multiple instances
nat1 = pat.NaTType()
nat2 = pat.NaTType()
print(f"\nTwo NaTType() calls create same object: {nat1 is nat2}")
print(f"nat1 == nat2: {nat1 == nat2}")

# Test in a Series
s = pd.Series([nat_from_call, pd.NaT, None])
print(f"\nSeries with [NaTType(), pd.NaT, None]:")
print(f"Series values: {s}")
print(f"Series.isna():\n{s.isna()}")

# Test NAType for comparison (it should work correctly)
na_from_call = pat.NAType()
print(f"\nFor comparison, NAType() behavior:")
print(f"na_from_call is pd.NA: {na_from_call is pd.NA}")
print(f"pd.isna(na_from_call): {pd.isna(na_from_call)}")