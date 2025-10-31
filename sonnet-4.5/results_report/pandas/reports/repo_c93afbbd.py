import numpy as np
import pandas as pd

categories = ["A", "B", "C"]
codes = np.array([-1, 0, 1, -1, 2], dtype=np.int64)

cat = pd.Categorical.from_codes(codes, categories=categories)
df = pd.DataFrame({"cat_col": cat})

print(f"Original null positions: {np.where(df['cat_col'].isna())[0]}")
print(f"Original values: {df['cat_col'].values}")

xchg = df.__dataframe__()
result = pd.api.interchange.from_dataframe(xchg)

print(f"Result null positions: {np.where(result['cat_col'].isna())[0]}")
print(f"Result values: {result['cat_col'].values}")

assert np.array_equal(df['cat_col'].isna().values, result['cat_col'].isna().values), \
    "Null positions changed during interchange!"