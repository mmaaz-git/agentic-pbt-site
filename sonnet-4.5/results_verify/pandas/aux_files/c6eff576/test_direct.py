import pyarrow as pa
from pandas.core.arrays.arrow import ArrowExtensionArray

arr = ArrowExtensionArray(pa.array([None]))
print(f"Array type: {arr._pa_array.type}")
print(f"Array: {arr}")

try:
    result = arr.insert(0, 42)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")