import pandas as pd

# Create empty DataFrame with specific dtypes  
df = pd.DataFrame({'a': pd.Series([], dtype='int64'), 'b': pd.Series([], dtype='float64')})
print("Original dtypes:", df.dtypes.to_dict())

# Test 'columns' orientation (default)
dict_columns = {'a': [], 'b': []}
result_columns = pd.DataFrame.from_dict(dict_columns, orient='columns')
print(f"\ncolumns orient: dtypes = {result_columns.dtypes.to_dict()}")

# Test 'index' orientation
dict_index = {}  # Empty dict for empty DataFrame
result_index = pd.DataFrame.from_dict(dict_index, orient='index')
print(f"\nindex orient: dtypes = {result_index.dtypes.to_dict()}")

# Test 'tight' orientation
dict_tight = df.to_dict(orient='tight')
result_tight = pd.DataFrame.from_dict(dict_tight, orient='tight')
print(f"\ntight orient: dtypes = {result_tight.dtypes.to_dict()}")
