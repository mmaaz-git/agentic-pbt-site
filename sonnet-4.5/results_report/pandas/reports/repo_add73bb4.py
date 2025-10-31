import pandas as pd

# Primary test case - the minimal failing example
s = pd.Series(['0'])
result = s.str.slice_replace(start=1, stop=0, repl='')
print(f"Result: {repr(result.iloc[0])}")

# What we expect based on Python slicing semantics
expected = '0'[:1] + '' + '0'[0:]
print(f"Expected: {repr(expected)}")

# Verify the bug
try:
    assert result.iloc[0] == expected
    print("Test PASSED")
except AssertionError:
    print(f"Test FAILED: Expected {repr(expected)}, got {repr(result.iloc[0])}")

print("\n--- Additional test cases demonstrating the bug ---\n")

# Test case 2: 'hello'.slice_replace(3, 1, 'X')
s2 = pd.Series(['hello'])
result2 = s2.str.slice_replace(start=3, stop=1, repl='X')
expected2 = 'hello'[:3] + 'X' + 'hello'[1:]
print(f"Test 2: 'hello'.slice_replace(3, 1, 'X')")
print(f"  Result: {repr(result2.iloc[0])}")
print(f"  Expected: {repr(expected2)}")
print(f"  Match: {result2.iloc[0] == expected2}")

# Test case 3: 'abc'.slice_replace(2, 1, '')
s3 = pd.Series(['abc'])
result3 = s3.str.slice_replace(start=2, stop=1, repl='')
expected3 = 'abc'[:2] + '' + 'abc'[1:]
print(f"\nTest 3: 'abc'.slice_replace(2, 1, '')")
print(f"  Result: {repr(result3.iloc[0])}")
print(f"  Expected: {repr(expected3)}")
print(f"  Match: {result3.iloc[0] == expected3}")

# Test case 4: 'test'.slice_replace(4, 2, 'XX')
s4 = pd.Series(['test'])
result4 = s4.str.slice_replace(start=4, stop=2, repl='XX')
expected4 = 'test'[:4] + 'XX' + 'test'[2:]
print(f"\nTest 4: 'test'.slice_replace(4, 2, 'XX')")
print(f"  Result: {repr(result4.iloc[0])}")
print(f"  Expected: {repr(expected4)}")
print(f"  Match: {result4.iloc[0] == expected4}")