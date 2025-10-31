from scipy.io.arff._arffread import split_data_line

print("Testing split_data_line with empty string...")
try:
    result, dialect = split_data_line('')
    print(f"Result: {result}, Dialect: {dialect}")
except IndexError as e:
    print(f"IndexError caught: {e}")
except Exception as e:
    print(f"Other exception: {type(e).__name__}: {e}")