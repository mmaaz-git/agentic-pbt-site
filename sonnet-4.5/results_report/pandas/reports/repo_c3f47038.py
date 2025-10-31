from pandas.core.computation.parsing import clean_column_name

name_with_null = '\x00'
result = clean_column_name(name_with_null)
print(f"Result: {result}")