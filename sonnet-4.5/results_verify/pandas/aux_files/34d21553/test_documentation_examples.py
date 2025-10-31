import pandas as pd

# Test based on documentation examples
print("Testing based on documentation examples:")
print()

# Example from documentation
s = pd.Series(['a', 'ab', 'abc', 'abdc', 'abcde'])
print("Original series:", s.tolist())
print()

# "Specify just stop, meaning the start of the string to stop is replaced with repl, and the rest of the string is included"
result = s.str.slice_replace(stop=2, repl='X')
print("Documentation says for stop=2, repl='X':")
print("Expected: ['X', 'X', 'Xc', 'Xdc', 'Xcde']")
print("Got:     ", result.tolist())
print()

# Test with both start and stop as None
print("Testing with start=None, stop=None, repl='X':")
print("Based on docs:")
print("- start=None means 'slice from the start of the string'")
print("- stop=None means 'slice until the end of the string'")
print("So slice_replace(None, None, 'X') should replace the entire string with 'X'")
result = s.str.slice_replace(start=None, stop=None, repl='X')
print("Result:", result.tolist())
print()

# But what about when repl is empty?
print("Testing with start=None, stop=None, repl='':")
print("This should replace the entire string with an empty string")
print("But the bug report claims it should be: original[:None] + '' + original[None:]")
result = s.str.slice_replace(start=None, stop=None, repl='')
print("Result:", result.tolist())
print()

# Let's understand Python slicing with None
test_str = "hello"
print("Understanding Python slicing with None:")
print(f"'hello'[None:None] = {test_str[None:None]!r}")
print(f"'hello'[:None] = {test_str[:None]!r}")
print(f"'hello'[None:] = {test_str[None:]!r}")
print(f"'hello'[:] = {test_str[:]!r}")