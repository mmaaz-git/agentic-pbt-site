import pandas as pd
import pandas.api.typing as pat

nat_from_call = pat.NaTType()

print(f"nat_from_call is pd.NaT: {nat_from_call is pd.NaT}")
print(f"pd.isna(nat_from_call): {pd.isna(nat_from_call)}")

s = pd.Series([nat_from_call, pd.NaT])
print(f"\nSeries.isna():\n{s.isna()}")

# Let's do some additional testing
print("\n--- Additional Testing ---")
print(f"Type of nat_from_call: {type(nat_from_call)}")
print(f"Type of pd.NaT: {type(pd.NaT)}")
print(f"Are types same?: {type(nat_from_call) == type(pd.NaT)}")

# Test multiple instances
inst1 = pat.NaTType()
inst2 = pat.NaTType()
print(f"\nAre two NaTType() calls the same object?: {inst1 is inst2}")
print(f"Are they equal?: {inst1 == inst2}")

# Compare with NAType for consistency
na_from_call = pat.NAType()
print(f"\nFor comparison - na_from_call is pd.NA: {na_from_call is pd.NA}")
print(f"pd.isna(na_from_call): {pd.isna(na_from_call)}")