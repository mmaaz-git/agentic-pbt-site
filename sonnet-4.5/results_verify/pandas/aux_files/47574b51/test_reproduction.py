import pandas as pd

s = pd.Series(['a-b-c'])

split_result = s.str.split('-', regex=False)
print(f"split('-', regex=False): {split_result.iloc[0]}")

try:
    rsplit_result = s.str.rsplit('-', regex=False)
    print(f"rsplit('-', regex=False): {rsplit_result.iloc[0]}")
except TypeError as e:
    print(f"TypeError: {e}")

# Also test that split() supports regex=True
split_regex_result = s.str.split('-', regex=True)
print(f"split('-', regex=True): {split_regex_result.iloc[0]}")

# Test rsplit without regex parameter
rsplit_no_regex = s.str.rsplit('-')
print(f"rsplit('-'): {rsplit_no_regex.iloc[0]}")