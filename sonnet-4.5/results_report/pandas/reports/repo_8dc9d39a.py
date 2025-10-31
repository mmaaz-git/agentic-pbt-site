import pyarrow as pa
from pandas.core.arrays.arrow import ArrowExtensionArray

# Create an ArrowExtensionArray containing only None
arr = ArrowExtensionArray(pa.array([None]))

# Try to insert a non-None value
try:
    result = arr.insert(0, 42)
    print(f"Success: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")