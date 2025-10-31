import pandas as pd
import numpy as np
from pandas.api.interchange import from_dataframe

# Create a DataFrame with an integer that's outside int64 range
# int64 min is -9,223,372,036,854,775,808
# We use -9,223,372,036,854,775,809 (one less than min)
df = pd.DataFrame({'a': [-9_223_372_036_854_775_809]})

print(f"DataFrame dtype: {df['a'].dtype}")
print(f"Value is below int64 min: {-9_223_372_036_854_775_809 < np.iinfo(np.int64).min}")

# This succeeds
interchange_df = df.__dataframe__()
print("Created interchange object successfully")

# This fails with NotImplementedError
try:
    df_roundtrip = from_dataframe(interchange_df)
    print("Successfully converted from interchange")
except NotImplementedError as e:
    print(f"Failed with error: {e}")