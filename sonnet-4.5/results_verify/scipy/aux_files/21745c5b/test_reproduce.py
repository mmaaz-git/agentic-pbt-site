import pandas as pd

s = pd.Series(['a.b.c.d'])

print("Testing split() with regex=True (treats '.' as regex):")
print(s.str.split('.', regex=True).iloc[0])

print("\nTesting split() with regex=False (treats '.' as literal):")
print(s.str.split('.', regex=False).iloc[0])

print("\nTesting rsplit() without regex parameter:")
print(s.str.rsplit('.').iloc[0])

print("\nTrying to use rsplit() with regex=False:")
try:
    result = s.str.rsplit('.', regex=False)
    print(f"Result: {result.iloc[0]}")
except TypeError as e:
    print(f"ERROR: {e}")