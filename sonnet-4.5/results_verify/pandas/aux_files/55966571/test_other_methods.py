import pandas as pd

# Test different category manipulation methods to check consistency

# Original unordered categorical
cat = pd.Categorical(['c', 'b', 'a'], categories=['c', 'b', 'a', 'x', 'y'])
print("Original categories:", list(cat.categories))

# Test remove_unused_categories
cat_unused = pd.Categorical(['c', 'b'], categories=['c', 'b', 'a', 'x'])
result_unused = cat_unused.remove_unused_categories()
print("\nremove_unused_categories:")
print(f"  Before: {list(cat_unused.categories)}")
print(f"  After:  {list(result_unused.categories)}")

# Test add_categories
cat_add = pd.Categorical(['c', 'b', 'a'], categories=['c', 'b', 'a'])
result_add = cat_add.add_categories(['x', 'y'])
print("\nadd_categories:")
print(f"  Before: {list(cat_add.categories)}")
print(f"  After:  {list(result_add.categories)}")

# Test set_categories
cat_set = pd.Categorical(['c', 'b', 'a'], categories=['c', 'b', 'a', 'x'])
result_set = cat_set.set_categories(['c', 'b', 'a', 'y', 'z'])
print("\nset_categories:")
print(f"  Before: {list(cat_set.categories)}")
print(f"  After:  {list(result_set.categories)}")

# Test remove_categories (the problematic one)
cat_remove = pd.Categorical(['c', 'b', 'a'], categories=['c', 'b', 'a', 'x'])
result_remove = cat_remove.remove_categories(['x'])
print("\nremove_categories:")
print(f"  Before: {list(cat_remove.categories)}")
print(f"  After:  {list(result_remove.categories)}")
print(f"  SORTED: {list(result_remove.categories) == sorted(result_remove.categories)}")