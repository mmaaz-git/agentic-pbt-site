import numpy as np
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')
from pandas.core.array_algos.masked_reductions import sum as masked_sum
from pandas.core.nanops import check_below_min_count
from pandas._libs import missing as libmissing

print("Tracing execution for all-masked array with min_count=0")
print("=" * 60)

arr = np.array([1, 2, 3], dtype=np.int64)
mask = np.ones(3, dtype=bool)  # All True = all masked
min_count = 0

print(f"Array shape: {arr.shape}")
print(f"Mask: {mask}")
print(f"min_count: {min_count}")
print()

# Check what check_below_min_count returns
below = check_below_min_count(arr.shape, mask, min_count)
print(f"check_below_min_count returns: {below}")
print()

# With min_count=0, below is False because:
# - min_count is not > 0, so the function returns False
print("Since min_count=0, check_below_min_count returns False")
print("This means _reductions won't return NA early")
print()

# Now check with min_count=1
min_count2 = 1
below2 = check_below_min_count(arr.shape, mask, min_count2)
print(f"With min_count=1, check_below_min_count returns: {below2}")

# Calculate non_nulls manually
non_nulls = mask.size - mask.sum()
print(f"non_nulls = {mask.size} - {mask.sum()} = {non_nulls}")
print(f"Since non_nulls ({non_nulls}) < min_count ({min_count2}), returns True")