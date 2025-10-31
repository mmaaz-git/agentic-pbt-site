import pandas as pd
from pandas.core.interchange.from_dataframe import categorical_column_to_series

# Create categorical data with missing value (-1 is the sentinel for missing)
cat_data = pd.Categorical.from_codes([-1], categories=['a'], ordered=False)
series = pd.Series(cat_data, name="cat_col")

print(f"Original series value: {series.iloc[0]}")
print(f"Original is NaN: {pd.isna(series.iloc[0])}")

# Create interchange object and extract column
interchange_obj = pd.DataFrame({"cat_col": series}).__dataframe__(allow_copy=True)
col = interchange_obj.get_column_by_name("cat_col")

# Convert using the function that has the bug
result_series, _ = categorical_column_to_series(col)

print(f"After interchange value: {result_series.iloc[0]}")
print(f"After interchange is NaN: {pd.isna(result_series.iloc[0])}")

# Show the problem: NaN became a valid category
if pd.isna(series.iloc[0]) and not pd.isna(result_series.iloc[0]):
    print("\nBUG CONFIRMED: Missing value (NaN) was converted to a valid category!")