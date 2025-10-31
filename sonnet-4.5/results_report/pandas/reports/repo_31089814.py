from pandas.core.computation.parsing import clean_column_name

# Test with null byte character
result = clean_column_name('\x00')
print(f"Result: {result}")