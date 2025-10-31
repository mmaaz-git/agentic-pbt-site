import numpy as np
from pandas.core.array_algos import masked_reductions
from pandas._libs import missing as libmissing

values = np.array([0.])
mask = np.array([True])

print("Testing masked reductions with all values masked (mask=[True]):")
print("=" * 60)

sum_result = masked_reductions.sum(values, mask, skipna=True)
prod_result = masked_reductions.prod(values, mask, skipna=True)
mean_result = masked_reductions.mean(values, mask, skipna=True)
min_result = masked_reductions.min(values, mask, skipna=True)
max_result = masked_reductions.max(values, mask, skipna=True)
var_result = masked_reductions.var(values, mask, skipna=True)
std_result = masked_reductions.std(values, mask, skipna=True)

print(f"sum:  {sum_result} (type: {type(sum_result).__name__}) - Is NA? {sum_result is libmissing.NA}")
print(f"prod: {prod_result} (type: {type(prod_result).__name__}) - Is NA? {prod_result is libmissing.NA}")
print(f"mean: {mean_result} (type: {type(mean_result).__name__}) - Is NA? {mean_result is libmissing.NA}")
print(f"min:  {min_result} (type: {type(min_result).__name__}) - Is NA? {min_result is libmissing.NA}")
print(f"max:  {max_result} (type: {type(max_result).__name__}) - Is NA? {max_result is libmissing.NA}")
print(f"var:  {var_result} (type: {type(var_result).__name__}) - Is NA? {var_result is libmissing.NA}")
print(f"std:  {std_result} (type: {type(std_result).__name__}) - Is NA? {std_result is libmissing.NA}")

print("\nBUG SUMMARY:")
print("-" * 60)
print("sum and prod return numeric values (0.0 and 1.0) when all values are masked,")
print("while mean, min, max, var, and std correctly return NA.")
print("This is an inconsistency in the masked_reductions module.")