from pandas.core.computation.parsing import clean_column_name

try:
    result = clean_column_name('\x00')
    print(f"Result: {repr(result)}")
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")