import pandas as pd
import numpy as np

# Create a simple DataFrame
df = pd.DataFrame({'A': [1, 2, 3]})

# Create weights with a negative value
weights = pd.Series([-1, 0, 1])

# Try to sample with negative weights (this should raise an error)
try:
    result = df.sample(n=1, weights=weights)
except ValueError as e:
    print(f"ValueError: {e}")