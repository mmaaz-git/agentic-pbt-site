import pandas as pd

# Test case 1
s = pd.Series(['0'])
result = s.str.slice_replace(start=None, stop=None, repl='')
print(f"Test 1 - String '0':")
print(f"Result: {result.iloc[0]!r}")
print(f"Expected: {'0'[:None] + '' + '0'[None:]!r}")
print()

# Test case 2
s = pd.Series(['hello'])
result = s.str.slice_replace(start=None, stop=None, repl='')
print(f"Test 2 - String 'hello':")
print(f"Result: {result.iloc[0]!r}")
print(f"Expected: {'hello'[:None] + '' + 'hello'[None:]!r}")
print()

# Let's also test what Python does with None slicing
test_str = '0'
print("Python behavior with None slicing:")
print(f"'0'[:None] = {test_str[:None]!r}")
print(f"'0'[None:] = {test_str[None:]!r}")
print(f"'0'[:None] + '' + '0'[None:] = {test_str[:None] + '' + test_str[None:]!r}")
print()

# Test with replacement text
s = pd.Series(['hello'])
result = s.str.slice_replace(start=None, stop=None, repl='REPLACEMENT')
print(f"Test 3 - String 'hello' with replacement 'REPLACEMENT':")
print(f"Result: {result.iloc[0]!r}")
print(f"Expected: {'hello'[:None] + 'REPLACEMENT' + 'hello'[None:]!r}")