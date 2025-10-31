import pandas as pd
from pandas.api.interchange import from_dataframe

df = pd.DataFrame({"col": pd.array([True, False, None], dtype="boolean")})
print("Original:", df["col"].tolist())

result = from_dataframe(df.__dataframe__())
print("After round-trip:", result["col"].tolist())