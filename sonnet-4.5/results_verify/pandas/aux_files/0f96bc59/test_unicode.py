# Test Python's standard upper() behavior with Unicode
test_cases = ['ß', 'ﬁ', 'ﬂ', 'ﬆ', 'ﬀ']

print("Python's standard str.upper() behavior:")
for char in test_cases:
    upper = char.upper()
    print(f"  '{char}' (len={len(char)}) -> '{upper}' (len={len(upper)})")

print("\nPython's str.capitalize() behavior:")
for test in ['ß', 'ßeta', 'ﬁle']:
    result = test.capitalize()
    print(f"  '{test}' -> '{result}'")