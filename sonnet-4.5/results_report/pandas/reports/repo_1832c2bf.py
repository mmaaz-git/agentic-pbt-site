import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})

try:
    df.rolling(window=2, method='invalid_method')
except ValueError as e:
    print(f"Error message: {e}")
    error_str = str(e)
    quote_count = error_str.count("'")
    print(f"Number of single quotes in error: {quote_count}")