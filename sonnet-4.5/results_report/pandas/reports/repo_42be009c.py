import numpy as np
import pandas as pd
from pandas.api.extensions import take

# Create test data
index = pd.Index([10.0, 20.0, 30.0])
arr = np.array([10.0, 20.0, 30.0])

# Test with allow_fill=True and fill_value=None
# According to documentation, -1 should be filled with NaN
print("Testing with allow_fill=True, fill_value=None:")
print("=" * 50)

index_result = take(index, [0, -1, 2], allow_fill=True, fill_value=None)
array_result = take(arr, [0, -1, 2], allow_fill=True, fill_value=None)

print(f"Index result: {list(index_result)}")
print(f"Array result: {list(array_result)}")
print()

# Check the behavior
print("Checking behavior at position 1 (index -1):")
print(f"Array result[1] is NaN: {pd.isna(array_result[1])}")
print(f"Index result[1] is NaN: {pd.isna(index_result[1])}")
print(f"Index result[1] value: {index_result[1]}")
print()

if pd.isna(array_result[1]) and not pd.isna(index_result[1]):
    print(f"BUG CONFIRMED: Index returns {index_result[1]} instead of NaN at position 1")
    print("The Index incorrectly treats -1 as a negative index (last element)")
    print("instead of as a missing value indicator.")
else:
    print("Bug not reproduced")