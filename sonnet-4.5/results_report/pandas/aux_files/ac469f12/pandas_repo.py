import pandas as pd
from pandas.api.types import union_categoricals

# Test Case 1: Primary Bug Scenario
print("Test Case 1: Primary Bug Scenario")
print("=" * 50)
cat_a = pd.Categorical(['a'])
cat_b = pd.Categorical(['a', 'b\x00'])
cat_c = pd.Categorical(['b'])

print(f"cat_a: values={cat_a.tolist()}, categories={cat_a.categories.tolist()}")
print(f"cat_b: values={cat_b.tolist()}, categories={cat_b.categories.tolist()}")
print(f"cat_c: values={cat_c.tolist()}, categories={cat_c.categories.tolist()}")
print()

result = union_categoricals([cat_a, cat_b, cat_c])

print(f"Expected categories: ['a', 'b\\x00', 'b']")
print(f"Actual categories: {result.categories.tolist()}")
print(f"Expected values: ['a', 'a', 'b\\x00', 'b']")
print(f"Actual values: {result.tolist()}")
print()

# Test Case 2: Hypothesis failing input
print("\nTest Case 2: Hypothesis Failing Input")
print("=" * 50)
cat1 = pd.Categorical(['0'])
cat2 = pd.Categorical(['0', '1\x00'])
cat3 = pd.Categorical(['1'])

print(f"cat1: values={cat1.tolist()}, categories={cat1.categories.tolist()}")
print(f"cat2: values={cat2.tolist()}, categories={cat2.categories.tolist()}")
print(f"cat3: values={cat3.tolist()}, categories={cat3.categories.tolist()}")
print()

result2 = union_categoricals([cat1, cat2, cat3])

print(f"Expected categories: ['0', '1\\x00', '1']")
print(f"Actual categories: {result2.categories.tolist()}")
print(f"Expected values: ['0', '0', '1\\x00', '1']")
print(f"Actual values: {result2.tolist()}")
print()

# Test Case 3: Two Categoricals Only (should work correctly)
print("\nTest Case 3: Two Categoricals Only")
print("=" * 50)
result3 = union_categoricals([cat_b, cat_c])
print(f"Combining only cat_b and cat_c:")
print(f"Expected categories: ['a', 'b\\x00', 'b']")
print(f"Actual categories: {result3.categories.tolist()}")
print(f"Expected values: ['a', 'b\\x00', 'b']")
print(f"Actual values: {result3.tolist()}")
print()

# Test Case 4: Different Order
print("\nTest Case 4: Different Order")
print("=" * 50)
result4 = union_categoricals([cat_c, cat_b, cat_a])
print(f"Combining in reverse order [cat_c, cat_b, cat_a]:")
print(f"Expected categories: ['b', 'a', 'b\\x00'] (or all three)")
print(f"Actual categories: {result4.categories.tolist()}")
print(f"Expected values: ['b', 'a', 'b\\x00', 'a']")
print(f"Actual values: {result4.tolist()}")