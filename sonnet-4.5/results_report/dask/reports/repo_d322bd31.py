import numpy as np
import pandas as pd
from dask.dataframe.utils import check_matching_columns

# Create two dataframes with different column names
# meta has column '0', actual has column 'NaN'
meta = pd.DataFrame(columns=[0, 1, 2])
actual = pd.DataFrame(columns=[float('nan'), 1, 2])

print(f"meta.columns: {meta.columns.tolist()}")
print(f"actual.columns: {actual.columns.tolist()}")
print(f"Are columns equal? {meta.columns.equals(actual.columns)}")
print()

# Try to validate the columns - should raise ValueError but doesn't
try:
    check_matching_columns(meta, actual)
    print("No error raised - BUG CONFIRMED!")
    print("The function incorrectly treats NaN column as equivalent to 0")
except ValueError as e:
    print(f"ValueError raised as expected: {e}")