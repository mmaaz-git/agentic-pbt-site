import pandas as pd

# Test case 1: Basic example showing data loss
s1 = pd.Series(['abc'])
result1 = s1.str.slice_replace(start=2, stop=1, repl='X').iloc[0]
expected1 = 'abc'[:2] + 'X' + 'abc'[1:]  # Should be 'abXbc'
print(f"Test 1 - Basic example:")
print(f"  Input:    'abc'")
print(f"  Operation: slice_replace(start=2, stop=1, repl='X')")
print(f"  Result:   {result1!r}")
print(f"  Expected: {expected1!r}")
print(f"  Data lost: Character 'b' at position 1 is missing!")
print()

# Test case 2: The specific failing case from hypothesis
s2 = pd.Series(['0'])
result2 = s2.str.slice_replace(start=1, stop=0, repl='X').iloc[0]
expected2 = '0'[:1] + 'X' + '0'[0:]  # Should be '0X0'
print(f"Test 2 - Hypothesis failing case:")
print(f"  Input:    '0'")
print(f"  Operation: slice_replace(start=1, stop=0, repl='X')")
print(f"  Result:   {result2!r}")
print(f"  Expected: {expected2!r}")
print(f"  Data lost: Character '0' at position 0 is missing!")
print()

# Test case 3: Another example with longer string
s3 = pd.Series(['hello world'])
result3 = s3.str.slice_replace(start=7, stop=5, repl='[INSERTED]').iloc[0]
expected3 = 'hello world'[:7] + '[INSERTED]' + 'hello world'[5:]  # Should be 'hello w[INSERTED]world'
print(f"Test 3 - Longer string example:")
print(f"  Input:    'hello world'")
print(f"  Operation: slice_replace(start=7, stop=5, repl='[INSERTED]')")
print(f"  Result:   {result3!r}")
print(f"  Expected: {expected3!r}")
print(f"  Data lost: Characters 'wo' at positions 5-6 are missing!")
print()

# Test case 4: Normal case (start < stop) works correctly
s4 = pd.Series(['test'])
result4 = s4.str.slice_replace(start=1, stop=3, repl='X').iloc[0]
expected4 = 'test'[:1] + 'X' + 'test'[3:]  # Should be 'tXt'
print(f"Test 4 - Normal case (start < stop) - works correctly:")
print(f"  Input:    'test'")
print(f"  Operation: slice_replace(start=1, stop=3, repl='X')")
print(f"  Result:   {result4!r}")
print(f"  Expected: {expected4!r}")
print(f"  Correct:  {result4 == expected4}")