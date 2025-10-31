import pandas as pd
from io import StringIO

df = pd.DataFrame({'x': [1.0, 2.0], 'y': [3.0, 4.0]})
print(f"Original DataFrame:")
print(df)
print(f"\nOriginal dtypes: {df.dtypes.to_dict()}")

json_str = df.to_json(orient="split")
print(f"\nJSON representation: {json_str}")

df_recovered = pd.read_json(StringIO(json_str), orient="split")
print(f"\nRecovered DataFrame:")
print(df_recovered)
print(f"\nRecovered dtypes: {df_recovered.dtypes.to_dict()}")

print(f"\nAre dtypes equal? {df.dtypes.equals(df_recovered.dtypes)}")
print(f"Are DataFrames equal? {df.equals(df_recovered)}")