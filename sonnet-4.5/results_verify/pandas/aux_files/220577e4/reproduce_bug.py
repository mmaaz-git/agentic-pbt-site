import pandas as pd

print("Test case 1: start=1, stop=0, repl=''")
s = pd.Series(['0'])
result = s.str.slice_replace(start=1, stop=0, repl='')

print(f"Pandas result: {result.iloc[0]!r}")
expected = '0'[:1] + '' + '0'[0:]
print(f"Expected:      {expected!r}")
print(f"Match: {result.iloc[0] == expected}")

print("\nTest case 2: start=3, stop=1, repl='X'")
s2 = pd.Series(['hello'])
result2 = s2.str.slice_replace(start=3, stop=1, repl='X')

print(f"Pandas result: {result2.iloc[0]!r}")
expected2 = 'hello'[:3] + 'X' + 'hello'[1:]
print(f"Expected:      {expected2!r}")
print(f"Match: {result2.iloc[0] == expected2}")

# Let's also test what Python's standard slicing does
print("\n--- Python slicing behavior ---")
print("When start > stop:")
s = "hello"
print(f"'hello'[3:1] = {s[3:1]!r} (empty string)")
print(f"'hello'[:3] + 'X' + 'hello'[1:] = {'hello'[:3] + 'X' + 'hello'[1:]!r}")

s = "0"
print(f"'0'[1:0] = {s[1:0]!r} (empty string)")
print(f"'0'[:1] + '' + '0'[0:] = {'0'[:1] + '' + '0'[0:]!r}")