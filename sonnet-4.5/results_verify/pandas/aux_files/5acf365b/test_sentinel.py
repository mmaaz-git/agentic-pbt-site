import pandas as pd
import numpy as np

# Create a categorical with nulls
df = pd.DataFrame({"cat_col": pd.Categorical(['a', None, 'b', None])})

# Get the interchange representation
interchange_df = df.__dataframe__()

# Get the column
col = interchange_df.get_column_by_name("cat_col")

# Inspect null representation
print("Null type info:", col.describe_null)

# Get the buffers to see the actual codes
buffers = col.get_buffers()
codes_buff, codes_dtype = buffers["data"]

# Look at the raw codes
import pandas.core.interchange.from_dataframe as ifdf
codes = ifdf.buffer_to_ndarray(codes_buff, codes_dtype, offset=col.offset, length=col.size())

print("Raw codes array:", codes)
print("Categories:", col.describe_categorical["categories"]._col if hasattr(col.describe_categorical["categories"], "_col") else "No _col")

# Show the issue:
categories = np.array(col.describe_categorical["categories"]._col)
print("\nCategories:", categories)
print("Codes modulo len(categories):", codes % len(categories))
print("Expected null positions (where codes == -1):", codes == -1)