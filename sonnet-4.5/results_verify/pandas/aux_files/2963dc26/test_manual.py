import pandas as pd
import numpy as np
from pandas.api.interchange import from_dataframe

categories = ['a', 'b', 'c']
codes = np.array([0, 1, 2, -1, 0, 1], dtype='int8')

cat = pd.Categorical.from_codes(codes, categories=categories)
df = pd.DataFrame({'cat': cat})

print("Original DataFrame:")
print(df)
print(f"Original NaN count: {df.isna().sum().sum()}")
print(f"Original values: {df['cat'].to_list()}")

result = from_dataframe(df.__dataframe__())

print("\nAfter interchange:")
print(result)
print(f"Result NaN count: {result.isna().sum().sum()}")
print(f"Result values: {result['cat'].to_list()}")