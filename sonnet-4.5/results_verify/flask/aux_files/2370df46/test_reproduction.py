import pandas as pd
from pandas.api.typing import NaTType, NAType

nat1 = NaTType()
nat2 = NaTType()
print(f"NaTType() is NaTType(): {nat1 is nat2}")
print(f"NaTType() is pd.NaT: {nat1 is pd.NaT}")

na1 = NAType()
na2 = NAType()
print(f"NAType() is NAType(): {na1 is na2}")
print(f"NAType() is pd.NA: {na1 is pd.NA}")

# Additional tests to understand the behavior
print("\nAdditional debugging:")
print(f"nat1 type: {type(nat1)}")
print(f"pd.NaT type: {type(pd.NaT)}")
print(f"nat1 == pd.NaT: {nat1 == pd.NaT}")
print(f"nat1 id: {id(nat1)}")
print(f"nat2 id: {id(nat2)}")
print(f"pd.NaT id: {id(pd.NaT)}")