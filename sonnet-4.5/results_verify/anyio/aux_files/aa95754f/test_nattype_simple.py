import pandas as pd
import pandas.api.typing as pat

nat1 = pat.NaTType()
nat2 = pat.NaTType()
na1 = pat.NAType()
na2 = pat.NAType()

print(f"NAType() is singleton: {na1 is na2}")
print(f"NAType() returns pd.NA: {na1 is pd.NA}")

print(f"NaTType() is singleton: {nat1 is nat2}")
print(f"NaTType() returns pd.NaT: {nat1 is pd.NaT}")

# Additional checks
print("\nAdditional checks:")
print(f"nat1 == nat2: {nat1 == nat2}")
print(f"nat1 == pd.NaT: {nat1 == pd.NaT}")
print(f"nat1 is pd.NaT: {nat1 is pd.NaT}")
print(f"isinstance(nat1, pat.NaTType): {isinstance(nat1, pat.NaTType)}")
print(f"isinstance(pd.NaT, pat.NaTType): {isinstance(pd.NaT, pat.NaTType)}")
print(f"type(nat1): {type(nat1)}")
print(f"type(pd.NaT): {type(pd.NaT)}")