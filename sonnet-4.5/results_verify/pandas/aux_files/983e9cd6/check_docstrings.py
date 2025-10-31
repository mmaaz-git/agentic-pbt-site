import pandas as pd
from pandas.io.json import read_json, to_json

print("=== read_json docstring ===")
print(read_json.__doc__)
print("\n" + "="*50 + "\n")
print("=== to_json docstring ===")
print(to_json.__doc__)
print("\n" + "="*50 + "\n")
print("=== DataFrame.to_json docstring ===")
print(pd.DataFrame.to_json.__doc__)