import pandas as pd

df = pd.DataFrame({'a': [1.5], 'b': [2]})

print("Original dtypes:")
print(df.dtypes)
print()

print("After T.T dtypes:")
print(df.T.T.dtypes)
print()

print("Testing if dtypes are preserved:")
try:
    assert df['b'].dtype == df.T.T['b'].dtype
    print("PASS: dtypes are preserved")
except AssertionError:
    print("FAIL: dtype of column 'b' changed from {} to {}".format(df['b'].dtype, df.T.T['b'].dtype))