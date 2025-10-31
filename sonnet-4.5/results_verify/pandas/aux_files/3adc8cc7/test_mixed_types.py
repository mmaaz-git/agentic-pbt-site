import pandas as pd

# Test with different types that can't be coerced
df1 = pd.DataFrame({'a': [1], 'b': ['hello']})
print("Test 1 - int and string:")
print("Original:", df1.dtypes.to_dict())
print("After T:", df1.T.dtypes.to_dict())
print("After T.T:", df1.T.T.dtypes.to_dict())
print()

# Test with int and float
df2 = pd.DataFrame({'a': [1], 'b': [2.5]})
print("Test 2 - int and float:")
print("Original:", df2.dtypes.to_dict())
print("After T:", df2.T.dtypes.to_dict())
print("After T.T:", df2.T.T.dtypes.to_dict())
print()

# Test with multiple ints and floats
df3 = pd.DataFrame({'a': [1, 2], 'b': [3.5, 4.5], 'c': [5, 6]})
print("Test 3 - multiple rows, mixed int/float:")
print("Original:", df3.dtypes.to_dict())
print("After T:", df3.T.dtypes.to_dict())
print("After T.T:", df3.T.T.dtypes.to_dict())
