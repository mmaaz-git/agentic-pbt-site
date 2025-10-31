import pandas as pd

# Check docstring for pd.NA
print("pd.NA docstring:")
print("-" * 60)
print(pd.NA.__doc__)
print()

# Check class docstring
print("NAType class docstring:")
print("-" * 60)
print(pd.api.typing.NAType.__doc__)
print()

# Check __eq__ method docstring
print("NAType.__eq__ docstring:")
print("-" * 60)
print(pd.api.typing.NAType.__eq__.__doc__)
print()

# Check available methods
print("NAType methods:")
print("-" * 60)
for method in dir(pd.NA):
    if not method.startswith('_'):
        print(f"  {method}")