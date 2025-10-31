import pandas as pd

# Test case 1: Primary test case from bug report
s = pd.Series(['abc'])
result = s.str.slice_replace(start=2, stop=1, repl='X').iloc[0]
expected = 'abc'[:2] + 'X' + 'abc'[1:]

print("Test Case 1: strings=['abc'], start=2, stop=1")
print(f"Result:   {result!r}")
print(f"Expected: {expected!r}")
print(f"Data loss: Character 'b' was {'deleted' if 'b' not in result else 'preserved'}")
print()

# Test case 2: Negative index test
s2 = pd.Series(['hello'])
result2 = s2.str.slice_replace(start=-1, stop=-3, repl='X').iloc[0]
# For 'hello': -1 is index 4 ('o'), -3 is index 2 ('l')
# Since -1 > -3 in terms of actual indices (4 > 2), we have start > stop
expected2 = 'hello'[:4] + 'X' + 'hello'[2:]

print("Test Case 2: strings=['hello'], start=-1, stop=-3")
print(f"Result:   {result2!r}")
print(f"Expected: {expected2!r}")
print(f"Data loss: Substring 'll' was {'deleted' if 'll' not in result2 else 'preserved'}")
print()

# Test case 3: Larger gap test
s3 = pd.Series(['0123456789'])
result3 = s3.str.slice_replace(start=5, stop=3, repl='X').iloc[0]
expected3 = '0123456789'[:5] + 'X' + '0123456789'[3:]

print("Test Case 3: strings=['0123456789'], start=5, stop=3")
print(f"Result:   {result3!r}")
print(f"Expected: {expected3!r}")
print(f"Data loss: Substring '34' was {'deleted' if '34' not in result3 else 'preserved'}")
print()

# Test case 4: Normal case (control) - should work correctly
s4 = pd.Series(['test'])
result4 = s4.str.slice_replace(start=1, stop=3, repl='X').iloc[0]
expected4 = 'test'[:1] + 'X' + 'test'[3:]

print("Test Case 4 (Control): strings=['test'], start=1, stop=3")
print(f"Result:   {result4!r}")
print(f"Expected: {expected4!r}")
print(f"Correct:  {result4 == expected4}")