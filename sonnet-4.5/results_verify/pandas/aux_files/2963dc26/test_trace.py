import pandas as pd
import numpy as np

categories = ['a', 'b', 'c']
codes = np.array([0, 1, 2, -1, 0, 1], dtype='int8')

print("Original codes:", codes)
print("What happens with modulo operation:")
print(f"  -1 % 3 = {-1 % 3}")
print()

# This is what the categorical_column_to_series function does
if len(categories) > 0:
    values = np.array(categories)[codes % len(categories)]
    print(f"Values after modulo: {values}")
    print("Notice that -1 % 3 = 2, so -1 becomes 'c'!")

# Create a Series to show what the bug produces
cat = pd.Categorical(values, categories=categories)
data = pd.Series(cat)
print("\nResulting Series (before set_nulls):")
print(data)