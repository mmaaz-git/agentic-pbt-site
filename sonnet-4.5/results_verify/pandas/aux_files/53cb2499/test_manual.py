import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
from pandas.api.interchange import from_dataframe

df = pd.DataFrame({'col': pd.array([True, False, None], dtype='boolean')})
print("Original:")
print(df)
print("Has NA?", df['col'].isna().any())
print("Dtype:", df['col'].dtype)

result = from_dataframe(df.__dataframe__())
print("\nAfter round-trip:")
print(result)
print("Has NA?", result['col'].isna().any())
print("Dtype:", result['col'].dtype)