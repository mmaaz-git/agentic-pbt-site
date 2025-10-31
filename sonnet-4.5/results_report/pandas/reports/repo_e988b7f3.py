import pandas as pd
import numpy as np
from pandas.core.interchange.from_dataframe import from_dataframe

categories = pd.Index(['a', 'b', 'c'])
codes = np.array([0, 1, -1, 2], dtype=np.int8)

cat = pd.Categorical.from_codes(codes, categories=categories)
df_original = pd.DataFrame({'col': cat})

print(f"Original: {df_original['col'].values}")
print(f"Is null at index 2? {pd.isna(df_original['col'].iloc[2])}")

df_result = from_dataframe(df_original.__dataframe__())

print(f"Result: {df_result['col'].values}")
print(f"Is null at index 2? {pd.isna(df_result['col'].iloc[2])}")
print(f"Expected NaN at index 2, but got: '{df_result['col'].iloc[2]}'")