import pandas as pd

# Test case that crashes
s = pd.Series(['hello'])
result = s.str.slice_replace(start=1, stop=0, repl='X')

print(f"Input string: {s.iloc[0]}")
print(f"start=1, stop=0, repl='X'")
print(f"Result: {result.iloc[0]}")
print(f"Expected (s[:start] + repl + s[stop:]): {s.iloc[0][:1] + 'X' + s.iloc[0][0:]}")

# Verify the bug
expected = s.iloc[0][:1] + 'X' + s.iloc[0][0:]
actual = result.iloc[0]

if actual == expected:
    print("\nPASSED: Result matches expected behavior")
else:
    print(f"\nFAILED: Bug confirmed!")
    print(f"  Expected: '{expected}'")
    print(f"  Got:      '{actual}'")