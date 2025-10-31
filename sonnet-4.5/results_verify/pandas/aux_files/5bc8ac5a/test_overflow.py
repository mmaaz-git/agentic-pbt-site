import pandas as pd
from io import StringIO

print("Testing integer overflow bug in pandas.read_csv")
print("="*50)

csv = "value\n2147483648"
result = pd.read_csv(StringIO(csv), dtype={'value': 'int32'})

print(f"Input value:  2147483648")
print(f"int32 max:    2147483647")
print(f"Result value: {result['value'].iloc[0]}")
print(f"Result type:  {type(result['value'].iloc[0])}")

print("\nDataFrame:")
print(result)

print("\nChecking if overflow occurred:")
if result['value'].iloc[0] == -2147483648:
    print("BUG CONFIRMED: Value wrapped around to -2147483648")
else:
    print(f"Unexpected result: {result['value'].iloc[0]}")