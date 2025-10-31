import pandas as pd
import dask.dataframe as dd
import dask

# Test with default config
print("Default config value:", dask.config.get('dataframe.convert-string'))
df_pandas = pd.DataFrame({'text': ['hello', 'world']})
df_dask = dd.from_pandas(df_pandas, npartitions=1)
result = df_dask.compute()
print(f"With default config - Original: {df_pandas['text'].dtype}, After: {result['text'].dtype}")

# Test with convert-string=False
dask.config.set({'dataframe.convert-string': False})
print("\nConfig set to False:", dask.config.get('dataframe.convert-string'))
df_dask2 = dd.from_pandas(df_pandas, npartitions=1)
result2 = df_dask2.compute()
print(f"With convert-string=False - Original: {df_pandas['text'].dtype}, After: {result2['text'].dtype}")

# Test with convert-string=True
dask.config.set({'dataframe.convert-string': True})
print("\nConfig set to True:", dask.config.get('dataframe.convert-string'))
df_dask3 = dd.from_pandas(df_pandas, npartitions=1)
result3 = df_dask3.compute()
print(f"With convert-string=True - Original: {df_pandas['text'].dtype}, After: {result3['text'].dtype}")