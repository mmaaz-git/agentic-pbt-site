import pandas as pd

# Test with non-empty DataFrame
df_non_empty = pd.DataFrame({'a': [1, 2], 'b': [3.0, 4.0]})
print("Non-empty DataFrame test:")
print("Original:", df_non_empty.columns.tolist())

dict_non_empty = df_non_empty.to_dict(orient='index')
print("to_dict result:", dict_non_empty)

reconstructed_non_empty = pd.DataFrame.from_dict(dict_non_empty, orient='index')
print("Reconstructed:", reconstructed_non_empty.columns.tolist())
print("Equal?", df_non_empty.equals(reconstructed_non_empty))
print()

# Test with orient='tight' for empty DataFrame
df_empty = pd.DataFrame({'a': pd.Series([], dtype='int64'), 'b': pd.Series([], dtype='float64')})
print("Empty DataFrame with orient='tight':")
print("Original:", df_empty.columns.tolist())

tight_dict = df_empty.to_dict(orient='tight')
print("to_dict(orient='tight') result:", tight_dict)

reconstructed_tight = pd.DataFrame.from_dict(tight_dict, orient='tight')
print("Reconstructed:", reconstructed_tight.columns.tolist())
print("Equal?", df_empty.equals(reconstructed_tight))
print()

# Test with orient='split' for empty DataFrame
print("Empty DataFrame with orient='split':")
split_dict = df_empty.to_dict(orient='split')
print("to_dict(orient='split') result:", split_dict)

reconstructed_split = pd.DataFrame(split_dict['data'], index=split_dict['index'], columns=split_dict['columns'])
print("Reconstructed:", reconstructed_split.columns.tolist())
print("Equal?", df_empty.equals(reconstructed_split))