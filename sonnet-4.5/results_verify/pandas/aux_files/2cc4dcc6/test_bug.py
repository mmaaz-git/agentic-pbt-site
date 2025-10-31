import pandas as pd
from pandas.api.types import union_categoricals

# Reproduce the exact bug from the report
cat1 = pd.Categorical(['0'], categories=['0'])
cat2 = pd.Categorical(['0'], categories=['0'])
cat3 = pd.Categorical(['0\x00'], categories=['0\x00'])

print("Initial categoricals:")
print(f"cat1: values={list(cat1)}, categories={list(cat1.categories)}")
print(f"cat2: values={list(cat2)}, categories={list(cat2.categories)}")
print(f"cat3: values={list(cat3)}, categories={list(cat3.categories)}")
print()

# Test associativity: (cat1 ∪ cat2) ∪ cat3 vs cat1 ∪ (cat2 ∪ cat3)
print("Test 1: Left associative - union_categoricals([union_categoricals([cat1, cat2]), cat3])")
intermediate1 = union_categoricals([cat1, cat2])
print(f"intermediate1 = union_categoricals([cat1, cat2]): values={list(intermediate1)}, categories={list(intermediate1.categories)}")
result_left = union_categoricals([intermediate1, cat3])
print(f"result_left = union_categoricals([intermediate1, cat3]): values={list(result_left)}, categories={list(result_left.categories)}")
print()

print("Test 2: Right associative - union_categoricals([cat1, union_categoricals([cat2, cat3])])")
intermediate2 = union_categoricals([cat2, cat3])
print(f"intermediate2 = union_categoricals([cat2, cat3]): values={list(intermediate2)}, categories={list(intermediate2.categories)}")
result_right = union_categoricals([cat1, intermediate2])
print(f"result_right = union_categoricals([cat1, intermediate2]): values={list(result_right)}, categories={list(result_right.categories)}")
print()

print("Comparison:")
print(f"Left associative result:  {list(result_left)}")
print(f"Right associative result: {list(result_right)}")
print(f"Are they equal? {list(result_left) == list(result_right)}")
print()

# Check for NaN
print("Check for NaN in right associative result:")
for i, val in enumerate(result_right):
    print(f"  result_right[{i}] = {repr(val)}, is NaN? {pd.isna(val)}")