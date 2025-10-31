import pandas as pd

# Create empty DataFrame with specific dtypes
df = pd.DataFrame({'a': pd.Series([], dtype='int64'), 'b': pd.Series([], dtype='float64')})
print("Original dtypes:", df.dtypes.to_dict())

# Test different orientations
for orient in ['dict', 'list', 'series', 'split', 'records', 'index']:
    try:
        if orient == 'index' and df.empty:
            continue  # Skip index for empty df
        dict_repr = df.to_dict(orient=orient)
        if orient not in ['records', 'index']:  # These don't have from_dict support
            try:
                result = pd.DataFrame.from_dict(dict_repr, orient=orient)
                print(f"\n{orient}: dtypes after round-trip = {result.dtypes.to_dict()}")
            except Exception as e:
                print(f"\n{orient}: from_dict failed - {e}")
        else:
            print(f"\n{orient}: no from_dict support for this orient")
    except Exception as e:
        print(f"\n{orient}: to_dict failed - {e}")
