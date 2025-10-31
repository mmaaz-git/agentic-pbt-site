import pandas as pd
from pandas import RangeIndex
import numpy as np

ri = RangeIndex(0, 1, 1)
values = np.array([0])

result = ri._shallow_copy(values)

print(f"Input values: {values}")
print(f"Result type: {type(result).__name__}")
print(f"Expected: RangeIndex (for equally-spaced values)")
print(f"Actual: {type(result).__name__}")
print(f"Is RangeIndex: {isinstance(result, RangeIndex)}")