import pandas as pd
import numpy as np
from pandas.api.interchange import from_dataframe

df = pd.DataFrame({'a': [-9_223_372_036_854_775_809]})

print(f"DataFrame dtype: {df['a'].dtype}")
print(f"Value is below int64 min: {-9_223_372_036_854_775_809 < np.iinfo(np.int64).min}")

interchange_df = df.__dataframe__()
print("Created interchange object successfully")

try:
    df_roundtrip = from_dataframe(interchange_df)
    print("Round-trip succeeded!")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")