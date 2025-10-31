import pandas.api.typing as pat

na1 = pat.NAType()
na2 = pat.NAType()
print(f"NAType() is singleton: {na1 is na2}")

nat1 = pat.NaTType()
nat2 = pat.NaTType()
print(f"NaTType() is singleton: {nat1 is nat2}")

# Additional tests to understand the behavior
import pandas as pd
print(f"\nNaTType() is pd.NaT: {pat.NaTType() is pd.NaT}")
print(f"NAType() is pd.NA: {pat.NAType() is pd.NA}")

# Check hash and equality
nat3 = pat.NaTType()
nat4 = pat.NaTType()
print(f"\nnat3 == nat4: {nat3 == nat4}")
print(f"nat3 is nat4: {nat3 is nat4}")
print(f"hash(nat3) == hash(nat4): {hash(nat3) == hash(nat4)}")