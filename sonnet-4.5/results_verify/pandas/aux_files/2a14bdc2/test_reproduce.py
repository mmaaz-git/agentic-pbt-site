import pandas as pd

# Simple reproduction test
df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})

print("Testing invalid method parameter...")
try:
    df.rolling(window=2, method='invalid_method')
except ValueError as e:
    error_msg = str(e)
    print(f"Error message: {error_msg}")
    single_quote_count = error_msg.count("'")
    print(f"Number of single quotes: {single_quote_count}")
    print(f"Is quote count even? {single_quote_count % 2 == 0}")