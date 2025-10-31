import sys
import numpy as np

sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

import pandas.core.array_algos.masked_reductions as masked_reductions
from pandas._libs import missing as libmissing

print("Testing reproduction code from bug report:")
print("=" * 50)

arr = np.array([1.0, 2.0, 3.0])
mask_all = np.array([True, True, True])

sum_result = masked_reductions.sum(arr, mask_all, skipna=True, min_count=0)
prod_result = masked_reductions.prod(arr, mask_all, skipna=True, min_count=0)
min_result = masked_reductions.min(arr, mask_all, skipna=True)
mean_result = masked_reductions.mean(arr, mask_all, skipna=True)

print(f"sum:  {sum_result}")
print(f"prod: {prod_result}")
print(f"min:  {min_result}")
print(f"mean: {mean_result}")

print("\n" + "=" * 50)
print("Testing with min_count=1:")
print("=" * 50)

sum_result_mc1 = masked_reductions.sum(arr, mask_all, skipna=True, min_count=1)
prod_result_mc1 = masked_reductions.prod(arr, mask_all, skipna=True, min_count=1)

print(f"sum (min_count=1):  {sum_result_mc1}")
print(f"prod (min_count=1): {prod_result_mc1}")

print("\n" + "=" * 50)
print("Checking if results match documentation:")
print("=" * 50)

# According to documentation, with min_count=0, sum should return 0 and prod should return 1
print(f"sum returns 0.0 with min_count=0: {sum_result == 0.0}")
print(f"prod returns 1.0 with min_count=0: {prod_result == 1.0}")

# With min_count=1, they should return NA
print(f"sum returns NA with min_count=1: {sum_result_mc1 is libmissing.NA}")
print(f"prod returns NA with min_count=1: {prod_result_mc1 is libmissing.NA}")

# min and mean should return NA regardless
print(f"min returns NA: {min_result is libmissing.NA}")
print(f"mean returns NA: {mean_result is libmissing.NA}")