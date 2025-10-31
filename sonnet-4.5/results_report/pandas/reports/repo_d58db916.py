import pandas as pd
import pandas.api.typing as typing

# Test NaTType singleton behavior
nat1 = typing.NaTType()
nat2 = typing.NaTType()

print(f"nat1 is nat2: {nat1 is nat2}")
print(f"nat1 is pd.NaT: {nat1 is pd.NaT}")
print(f"nat1 == nat2: {nat1 == nat2}")
print(f"nat1 == pd.NaT: {nat1 == pd.NaT}")

# Test NAType singleton behavior for comparison
na1 = typing.NAType()
na2 = typing.NAType()

print(f"\nna1 is na2: {na1 is na2}")
print(f"na1 is pd.NA: {na1 is pd.NA}")
print(f"na1 == na2: {na1 == na2}")
print(f"na1 == pd.NA: {na1 == pd.NA}")

# Additional diagnostics
print(f"\nType of nat1: {type(nat1)}")
print(f"Type of pd.NaT: {type(pd.NaT)}")
print(f"id(nat1): {id(nat1)}")
print(f"id(nat2): {id(nat2)}")
print(f"id(pd.NaT): {id(pd.NaT)}")