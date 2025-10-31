import numpy as np
from pandas.core.array_algos.masked_reductions import (
    sum as masked_sum,
    prod as masked_prod,
    mean as masked_mean,
    var as masked_var,
    std as masked_std
)
from pandas._libs import missing as libmissing

# Create a simple array with all values masked
arr = np.array([1, 2, 3], dtype=np.int64)
mask = np.ones(3, dtype=bool)  # All values are masked (True = missing)

print("Testing reduction operations with all-masked array:")
print("=" * 50)
print(f"Array: {arr}")
print(f"Mask:  {mask} (all True = all values missing)")
print()

# Test sum
result_sum = masked_sum(arr, mask, skipna=True, min_count=0)
print(f"sum(all_masked, min_count=0):  {result_sum}")
print(f"  Expected: NA")
print(f"  Got: {result_sum}")
print(f"  Is NA?: {result_sum is libmissing.NA}")
print()

# Test prod
result_prod = masked_prod(arr, mask, skipna=True, min_count=0)
print(f"prod(all_masked, min_count=0): {result_prod}")
print(f"  Expected: NA")
print(f"  Got: {result_prod}")
print(f"  Is NA?: {result_prod is libmissing.NA}")
print()

# Test mean (for comparison)
result_mean = masked_mean(arr, mask, skipna=True)
print(f"mean(all_masked):               {result_mean}")
print(f"  Expected: NA")
print(f"  Got: {result_mean}")
print(f"  Is NA?: {result_mean is libmissing.NA}")
print()

# Test var (for comparison)
result_var = masked_var(arr, mask, skipna=True)
print(f"var(all_masked):                {result_var}")
print(f"  Expected: NA")
print(f"  Got: {result_var}")
print(f"  Is NA?: {result_var is libmissing.NA}")
print()

# Test std (for comparison)
result_std = masked_std(arr, mask, skipna=True)
print(f"std(all_masked):                {result_std}")
print(f"  Expected: NA")
print(f"  Got: {result_std}")
print(f"  Is NA?: {result_std is libmissing.NA}")
print()

print("=" * 50)
print("Summary: sum and prod return identity elements (0, 1)")
print("         while mean, var, std correctly return NA")
print("         This is inconsistent behavior!")