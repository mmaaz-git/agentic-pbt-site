from scipy.io.arff._arffread import split_data_line

try:
    result, dialect = split_data_line('')
    print(f"Result: {result}, Dialect: {dialect}")
except IndexError as e:
    print(f"IndexError: {e}")