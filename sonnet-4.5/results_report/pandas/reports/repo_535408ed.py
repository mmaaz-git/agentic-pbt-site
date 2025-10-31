import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import warnings

# Show all warnings
warnings.filterwarnings('always')

df = pd.DataFrame({
    'A': [1.0, 1.0, 1.0, 1.0],
    'B': [2.0, 3.0, 4.0, 5.0],
    'class': ['cat1', 'cat1', 'cat2', 'cat2']
})

print("DataFrame:")
print(df)
print("\nColumn A values:", df['A'].values)
print("Column A min:", df['A'].min())
print("Column A max:", df['A'].max())
print("Column A range (max - min):", df['A'].max() - df['A'].min())

fig, ax = plt.subplots()
try:
    result = pd.plotting.radviz(df, 'class', ax=ax)
    print("\nRadViz completed successfully")

    # Check if the normalization resulted in NaN values
    def normalize(series):
        a = min(series)
        b = max(series)
        return (series - a) / (b - a)

    normalized_A = normalize(df['A'])
    print("\nNormalized column A:", normalized_A)
    print("Contains NaN:", np.any(np.isnan(normalized_A)))

except Exception as e:
    print(f"\nError occurred: {type(e).__name__}: {e}")
finally:
    plt.close('all')