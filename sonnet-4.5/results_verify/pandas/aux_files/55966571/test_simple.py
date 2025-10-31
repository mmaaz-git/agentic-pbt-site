import pandas as pd

# Test case 1: Example from bug report
cat = pd.Categorical(['c', 'b', 'a'], categories=['c', 'b', 'a', 'x'])
result = cat.remove_categories(['x'])

print("Test Case 1:")
print(f'Original: {list(cat.categories)}')
print(f'After remove_categories: {list(result.categories)}')
print(f'Expected: ["c", "b", "a"]')
print(f'Match: {list(result.categories) == ["c", "b", "a"]}')

# Test case 2: The failing example from property test
cat2 = pd.Categorical(['00'], categories=['00', '0'])
cat2_added = cat2.add_categories(['NEW_CATEGORY_XYZ'])
cat2_removed = cat2_added.remove_categories(['NEW_CATEGORY_XYZ'])

print("\nTest Case 2 (from property test):")
print(f'Original: {list(cat2.categories)}')
print(f'After add then remove: {list(cat2_removed.categories)}')
print(f'Expected: ["00", "0"]')
print(f'Match: {list(cat2_removed.categories) == ["00", "0"]}')

# Test case 3: Check ordered categoricals
cat3 = pd.Categorical(['c', 'b', 'a'], categories=['c', 'b', 'a', 'x'], ordered=True)
result3 = cat3.remove_categories(['x'])

print("\nTest Case 3 (ordered categorical):")
print(f'Original (ordered): {list(cat3.categories)}')
print(f'After remove_categories: {list(result3.categories)}')
print(f'Expected: ["c", "b", "a"]')
print(f'Match: {list(result3.categories) == ["c", "b", "a"]}')