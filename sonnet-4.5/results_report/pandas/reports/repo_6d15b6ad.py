import numpy as np
from pandas import DataFrame
from pandas.core.sample import preprocess_weights

# Create a DataFrame with 5 rows and 3 columns
df = DataFrame(np.random.randn(5, 3))

# Create weights array with a negative value
weights = np.array([1.0, 2.0, -1.0, 3.0, 4.0])

# Try to preprocess weights with a negative value
try:
    preprocess_weights(df, weights, axis=0)
except ValueError as e:
    print(f"Error message: {e}")