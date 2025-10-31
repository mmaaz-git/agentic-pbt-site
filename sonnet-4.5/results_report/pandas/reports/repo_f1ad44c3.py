from pandas.core.computation.parsing import clean_column_name

# Test case that should return the name unmodified but instead crashes
result = clean_column_name('\x00')
print(f"Result: {result}")