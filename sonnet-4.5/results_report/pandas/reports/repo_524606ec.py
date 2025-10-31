import pandas as pd

# Create a DataFrame with surrogate character '\ud800'
df = pd.DataFrame({
    'int_col': [0],
    'float_col': [0.0],
    'str_col': ['\ud800']
})

print(f"DataFrame created successfully:")
print(df)
print(f"\nDataFrame dtypes:")
print(df.dtypes)

# Get the interchange object
interchange_obj = df.__dataframe__()
print(f"\nInterchange object created successfully: {interchange_obj}")

# Attempt to convert back using from_dataframe
try:
    result = pd.api.interchange.from_dataframe(interchange_obj)
    print(f"\nResult DataFrame:")
    print(result)
except Exception as e:
    print(f"\nError occurred: {type(e).__name__}: {e}")