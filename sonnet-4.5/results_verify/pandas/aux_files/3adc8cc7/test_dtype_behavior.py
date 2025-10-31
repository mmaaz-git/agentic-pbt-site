import pandas as pd
import numpy as np

# Test what happens with transpose
df = pd.DataFrame({'a': [1.5], 'b': [2]})
print("Original DataFrame:")
print(df)
print("Original dtypes:", df.dtypes.to_dict())

df_t = df.T
print("\nFirst transpose:")
print(df_t)
print("After first transpose dtypes:", df_t.dtypes.to_dict())

df_tt = df.T.T
print("\nSecond transpose (T.T):")
print(df_tt)
print("After second transpose dtypes:", df_tt.dtypes.to_dict())

# Check if dtypes are object
print("\nIs first transpose dtype 'object'?", df_t.dtypes[0] == np.dtype('O'))
