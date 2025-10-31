import pandas as pd
from pandas.api.types import union_categoricals

print("Test Case 1: The reported bug scenario")
print("="*50)
cat_a = pd.Categorical(['a'])
cat_b = pd.Categorical(['a', 'b\x00'])
cat_c = pd.Categorical(['b'])

print(f"cat_a values: {cat_a.tolist()}, categories: {cat_a.categories.tolist()}")
print(f"cat_b values: {cat_b.tolist()}, categories: {cat_b.categories.tolist()}")
print(f"cat_c values: {cat_c.tolist()}, categories: {cat_c.categories.tolist()}")

result = union_categoricals([cat_a, cat_b, cat_c])

print(f"\nResult categories: {result.categories.tolist()}")
print(f"Result values: {result.tolist()}")

print(f"\nExpected categories: ['a', 'b\\x00', 'b']")
print(f"Expected values: ['a', 'a', 'b\\x00', 'b']")

print("\n" + "="*50)
print("Test Case 2: Using the exact failing input from hypothesis")
print("="*50)
cat1 = pd.Categorical(['0'])
cat2 = pd.Categorical(['0', '1\x00'])
cat3 = pd.Categorical(['1'])

print(f"cat1 values: {cat1.tolist()}, categories: {cat1.categories.tolist()}")
print(f"cat2 values: {cat2.tolist()}, categories: {cat2.categories.tolist()}")
print(f"cat3 values: {cat3.tolist()}, categories: {cat3.categories.tolist()}")

result2 = union_categoricals([cat1, cat2, cat3])

print(f"\nResult categories: {result2.categories.tolist()}")
print(f"Result values: {result2.tolist()}")

print(f"\nExpected categories: ['0', '1\\x00', '1']")
print(f"Expected values: ['0', '0', '1\\x00', '1']")

print("\n" + "="*50)
print("Test Case 3: What if we combine just cat_b and cat_c?")
print("="*50)
result3 = union_categoricals([cat_b, cat_c])
print(f"Result categories: {result3.categories.tolist()}")
print(f"Result values: {result3.tolist()}")

print("\n" + "="*50)
print("Test Case 4: What if we change the order?")
print("="*50)
result4 = union_categoricals([cat_c, cat_b, cat_a])
print(f"Result categories: {result4.categories.tolist()}")
print(f"Result values: {result4.tolist()}")