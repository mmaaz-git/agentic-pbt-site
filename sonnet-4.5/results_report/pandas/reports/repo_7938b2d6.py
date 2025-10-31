import pandas as pd
from pandas.core.interchange.from_dataframe import from_dataframe

cat_data = pd.Categorical(['a', 'b', None, 'a'], categories=['a', 'b', 'c'])
df = pd.DataFrame({'cat': cat_data})

print("Original DataFrame:")
print(df)
print(f"Original categorical codes: {df['cat'].cat.codes.tolist()}")

result = from_dataframe(df.__dataframe__())

print("\nResult DataFrame:")
print(result)
print(f"Result categorical codes: {result['cat'].cat.codes.tolist()}")

print(f"\nOriginal nulls: {df['cat'].isna().tolist()}")
print(f"Result nulls: {result['cat'].isna().tolist()}")