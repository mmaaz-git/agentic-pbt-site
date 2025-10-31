import numpy as np
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')
from pandas.core.array_algos.masked_reductions import sum as masked_sum, prod as masked_prod, mean as masked_mean, var as masked_var, std as masked_std
from pandas._libs import missing as libmissing

print("Testing masked reductions with all values masked...")
print("=" * 60)

# Create array with all values masked
arr = np.array([1, 2, 3], dtype=np.int64)
mask = np.ones(3, dtype=bool)  # All True means all values are masked

print(f"Array: {arr}")
print(f"Mask: {mask} (all True = all values masked)")
print()

# Test sum
result_sum = masked_sum(arr, mask, skipna=True, min_count=0)
print(f"sum result:  {result_sum}")
print(f"sum type:    {type(result_sum)}")
print(f"is NA?       {result_sum is libmissing.NA}")
print()

# Test prod
result_prod = masked_prod(arr, mask, skipna=True, min_count=0)
print(f"prod result: {result_prod}")
print(f"prod type:   {type(result_prod)}")
print(f"is NA?       {result_prod is libmissing.NA}")
print()

# Test mean (for comparison)
result_mean = masked_mean(arr, mask, skipna=True)
print(f"mean result: {result_mean}")
print(f"mean type:   {type(result_mean)}")
print(f"is NA?       {result_mean is libmissing.NA}")
print()

# Test var (for comparison)
result_var = masked_var(arr, mask, skipna=True)
print(f"var result:  {result_var}")
print(f"var type:    {type(result_var)}")
print(f"is NA?       {result_var is libmissing.NA}")
print()

# Test std (for comparison)
result_std = masked_std(arr, mask, skipna=True)
print(f"std result:  {result_std}")
print(f"std type:    {type(result_std)}")
print(f"is NA?       {result_std is libmissing.NA}")
print()

print("=" * 60)
print("Testing with min_count > 0...")
print()

# Test with min_count=1 (should return NA since we have 0 valid values)
result_sum_mc1 = masked_sum(arr, mask, skipna=True, min_count=1)
print(f"sum with min_count=1:  {result_sum_mc1} (is NA? {result_sum_mc1 is libmissing.NA})")

result_prod_mc1 = masked_prod(arr, mask, skipna=True, min_count=1)
print(f"prod with min_count=1: {result_prod_mc1} (is NA? {result_prod_mc1 is libmissing.NA})")