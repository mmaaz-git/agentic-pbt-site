print("=== Testing Python slicing behavior when start > stop ===")

test_str = "hello"

# Test slicing with start > stop
print(f"String: '{test_str}'")
print(f"test_str[3:1] = '{test_str[3:1]}' (empty string)")
print(f"test_str[4:2] = '{test_str[4:2]}' (empty string)")
print()

# Test how slice replacement should work conceptually
def manual_slice_replace(s, start, stop, repl):
    """Manually implement slice replacement following Python semantics"""
    return s[:start] + repl + s[stop:]

print("=== Manual slice replacement (following Python semantics) ===")
test_cases = [
    ("hello", 3, 1, "X"),
    ("abc", 2, 1, ""),
    ("test", 4, 2, "XX"),
    ("0", 1, 0, "")
]

for text, start, stop, repl in test_cases:
    result = manual_slice_replace(text, start, stop, repl)
    print(f"'{text}'[:{start}] + '{repl}' + '{text}'[{stop}:] = '{result}'")