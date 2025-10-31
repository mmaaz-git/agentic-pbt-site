import pyarrow as pa
from pandas.core.arrays.arrow import ArrowExtensionArray

# Create an array with only None values
# PyArrow will infer this as null type
arr = ArrowExtensionArray(pa.array([None]))

print(f"Array type: {arr._pa_array.type}")
print(f"Array values: {arr}")

# Try to fill NA values with 999
# This will crash because null type can't hold non-null values
try:
    filled = arr.fillna(value=999)
    print(f"Filled array: {filled}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")