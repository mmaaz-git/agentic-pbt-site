import pandas as pd
from hypothesis import given, strategies as st, settings

# First, let's test the specific failing input mentioned
print("=== Testing specific failing input ===")
s = pd.Series(['0'])
result = s.str.slice_replace(start=1, stop=0, repl='')
print(f"Input: s = pd.Series(['0']), start=1, stop=0, repl=''")
print(f"Result: {result.iloc[0]!r}")

# Calculate expected based on Python slicing semantics
expected = '0'[:1] + '' + '0'[0:]
print(f"Expected (based on orig[:start] + repl + orig[stop:]): {expected!r}")
print(f"Match: {result.iloc[0] == expected}")
print()

# Test additional examples from the bug report
print("=== Testing additional examples ===")
test_cases = [
    ('hello', 3, 1, 'X'),
    ('abc', 2, 1, ''),
    ('test', 4, 2, 'XX')
]

for text, start, stop, repl in test_cases:
    s = pd.Series([text])
    result = s.str.slice_replace(start=start, stop=stop, repl=repl)
    expected = text[:start] + repl + text[stop:]
    print(f"Input: '{text}', start={start}, stop={stop}, repl='{repl}'")
    print(f"Pandas result: {result.iloc[0]!r}")
    print(f"Expected: {expected!r}")
    print(f"Match: {result.iloc[0] == expected}")
    print()

# Now run the property-based test
print("=== Running property-based test ===")
failures_found = []

@given(st.lists(st.text(min_size=1), min_size=1), st.integers(min_value=-10, max_value=10), st.integers(min_value=-10, max_value=10), st.text(max_size=5))
@settings(max_examples=100)  # Reduced for faster testing
def test_slice_replace_consistency_with_python(strings, start, stop, repl):
    s = pd.Series(strings)
    replaced = s.str.slice_replace(start, stop, repl)
    for orig, repl_result in zip(s, replaced):
        if pd.notna(orig):
            expected = orig[:start] + repl + orig[stop:]
            assert repl_result == expected, f"Failed: orig={orig!r}, start={start}, stop={stop}, repl={repl!r}, got={repl_result!r}, expected={expected!r}"

try:
    test_slice_replace_consistency_with_python()
    print("Property test passed all examples")
except AssertionError as e:
    print(f"Property test found failures: {e}")