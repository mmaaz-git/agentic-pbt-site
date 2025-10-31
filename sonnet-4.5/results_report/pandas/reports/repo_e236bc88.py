import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pyarrow as pa
from pandas.core.arrays.arrow import ArrowExtensionArray

# Create an array with null type (happens when all values are None)
arr = ArrowExtensionArray(pa.array([None]))
print(f"Array type: {arr._pa_array.type}")
print(f"Array contents: {arr._pa_array}")

# Try to fill with a value
result = arr.fillna(0)
print(f"Result: {result}")