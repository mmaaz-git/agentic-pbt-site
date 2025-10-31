import pandas as pd

# Simple reproduction case from bug report
cat = pd.Categorical([], categories=['z', 'y', 'x'])
print(f"Initial categories: {list(cat.categories)}")

cat_removed = cat.remove_categories(['x'])
print(f"After remove_categories(['x']): {list(cat_removed.categories)}")

print(f"\nExpected: ['z', 'y']")
print(f"Actual: {list(cat_removed.categories)}")

# Test with ordered categorical
print("\n--- Testing with ordered=True ---")
cat_ordered = pd.Categorical([], categories=['z', 'y', 'x'], ordered=True)
print(f"Initial categories (ordered): {list(cat_ordered.categories)}")

cat_ordered_removed = cat_ordered.remove_categories(['x'])
print(f"After remove_categories(['x']) (ordered): {list(cat_ordered_removed.categories)}")

# Additional test: removing middle category
print("\n--- Testing removal of middle category ---")
cat2 = pd.Categorical([], categories=['z', 'y', 'x'])
cat2_removed = cat2.remove_categories(['y'])
print(f"Initial: ['z', 'y', 'x']")
print(f"After removing 'y': {list(cat2_removed.categories)}")
print(f"Expected: ['z', 'x']")