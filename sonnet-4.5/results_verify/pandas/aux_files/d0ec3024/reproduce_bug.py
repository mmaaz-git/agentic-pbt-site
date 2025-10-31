import pandas as pd
import numpy as np

# Test 1: Simple reproduction to see the error message
df = pd.DataFrame({'A': [1, 2, 3]})
weights = pd.Series([-1, 0, 1])

try:
    df.sample(n=1, weights=weights)
except ValueError as e:
    print("Error message from pandas:")
    print(e)
    print()
    print("Expected: 'weight vector may not include negative values'")
    print("Actual:   '" + str(e) + "'")
    print()
    if "many not" in str(e):
        print("BUG CONFIRMED: Typo 'many' instead of 'may' found in error message")
    else:
        print("Bug not confirmed - error message is correct")