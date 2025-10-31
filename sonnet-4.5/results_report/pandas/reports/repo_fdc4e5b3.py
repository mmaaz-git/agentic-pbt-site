import pandas as pd

# Simple example demonstrating the bug
cat = pd.Categorical(['c', 'b', 'a'], categories=['c', 'b', 'a', 'x'])
result = cat.remove_categories(['x'])

print(f'Original categories: {list(cat.categories)}')
print(f'After remove_categories: {list(result.categories)}')

# This assertion will fail - categories get sorted alphabetically
assert list(result.categories) == ['c', 'b', 'a'], f"Expected ['c', 'b', 'a'] but got {list(result.categories)}"