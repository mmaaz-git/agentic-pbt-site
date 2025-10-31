import pandas as pd
import numpy as np

# Create a categorical with missing values
categories = ['a', 'b', 'c']
codes = np.array([0, 1, 2, -1, 0, 1], dtype='int8')

cat = pd.Categorical.from_codes(codes, categories=categories)
df = pd.DataFrame({'cat': cat})

# Get the interchange object
interchange_df = df.__dataframe__()

# Get the column
col = interchange_df.get_column_by_name('cat')

# Check the null representation
null_kind, sentinel_val = col.describe_null
print(f"Null kind: {null_kind}")
print(f"Sentinel value: {sentinel_val}")

# Check buffers
buffers = col.get_buffers()
print(f"Validity buffer: {buffers['validity']}")

# Get the codes from the buffer
from pandas.core.interchange.from_dataframe import buffer_to_ndarray
from pandas.core.interchange.dataframe_protocol import DtypeKind

codes_buff, codes_dtype = buffers["data"]
codes = buffer_to_ndarray(
    codes_buff, codes_dtype, offset=col.offset, length=col.size()
)
print(f"Raw codes from buffer: {codes}")
print(f"Expected codes: [0, 1, 2, -1, 0, 1]")