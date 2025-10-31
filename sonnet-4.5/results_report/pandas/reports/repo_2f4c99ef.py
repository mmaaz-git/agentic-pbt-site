import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pyarrow as pa
from pandas.core.arrays.arrow import ArrowExtensionArray

# Create an ArrowExtensionArray with only None values
arr = ArrowExtensionArray(pa.array([None]))

# Try to fill NA values with 0
# This should raise an ArrowInvalid error instead of being caught properly
result = arr.fillna(0)
print(result)