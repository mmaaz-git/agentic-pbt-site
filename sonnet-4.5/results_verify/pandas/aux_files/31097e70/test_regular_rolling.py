import pandas as pd
import numpy as np

# Test what happens with regular pandas rolling with negative window
df = pd.DataFrame({'B': [0, 1, 2, 3, 4]})

try:
    result = df.rolling(window=-1).sum()
    print("Regular rolling accepts negative window:", result)
except Exception as e:
    print(f"Regular rolling with negative window raises: {type(e).__name__}: {e}")