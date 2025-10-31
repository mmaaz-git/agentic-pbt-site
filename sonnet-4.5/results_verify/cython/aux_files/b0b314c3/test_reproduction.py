#!/usr/bin/env python3

from hypothesis import given, strategies as st, assume
from Cython.Build.Dependencies import parse_list

# Test from the bug report
@given(st.lists(st.text(alphabet=st.characters(blacklist_categories=('Cs',)), min_size=1)))
def test_parse_list_bracket_delimited(items):
    assume(all(item.strip() for item in items))
    assume(all(',' not in item and '"' not in item and "'" not in item for item in items))
    s = '[' + ', '.join(items) + ']'
    result = parse_list(s)
    assert result == items, f"Expected {items}, got {result}"

# Manual test
print("Manual test cases:")

# Test case 1: The failing input from the report
print("\n1. Testing items=['#']:")
result = parse_list('[#]')
print(f"   Result: {result}")
print(f"   Expected: ['#']")
print(f"   Match: {result == ['#']}")

# Test case 2: foo#bar
print("\n2. Testing '[foo#bar]':")
result = parse_list('[foo#bar]')
print(f"   Result: {result}")
print(f"   Expected: ['foo#bar']")
print(f"   Match: {result == ['foo#bar']}")

# Test case 3: Multiple items with hash
print("\n3. Testing '[libA, libB#version]':")
result = parse_list('[libA, libB#version]')
print(f"   Result: {result}")
print(f"   Expected: ['libA', 'libB#version']")
print(f"   Match: {result == ['libA', 'libB#version']}")

# Test case 4: Space-separated with hash
print("\n4. Testing 'foo#bar baz':")
result = parse_list('foo#bar baz')
print(f"   Result: {result}")
print(f"   Expected: ['foo#bar', 'baz']")
print(f"   Match: {result == ['foo#bar', 'baz']}")

# Run the hypothesis test with the specific failing input
print("\n5. Running hypothesis test with ['#']:")
try:
    items = ['#']
    s = '[' + ', '.join(items) + ']'
    result = parse_list(s)
    assert result == items
    print("   Hypothesis test passed!")
except AssertionError as e:
    print(f"   Hypothesis test failed: Expected {items}, got {result}")

# Test normal cases without hash
print("\n6. Testing normal cases:")
print("   '[a, b, c]':", parse_list('[a, b, c]'))
print("   'a b c':", parse_list('a b c'))
print("   '[lib1, lib2]':", parse_list('[lib1, lib2]'))

# Test cases with quoted strings
print("\n7. Testing quoted strings:")
print('   \'["a b", c]\':', parse_list('["a b", c]'))
print("   \"['x y', z]\":", parse_list("['x y', z]"))

# Verify the documented behavior from docstring
print("\n8. Testing docstring examples:")
print('   parse_list(""):', parse_list(""))
print('   parse_list("a"):', parse_list("a"))
print('   parse_list("a b c"):', parse_list("a b c"))
print('   parse_list("[a, b, c]"):', parse_list("[a, b, c]"))
print('   parse_list(\'a " " b\'):', parse_list('a " " b'))
print('   parse_list(\'[a, ",a", "a,", ",", ]\'):', parse_list('[a, ",a", "a,", ",", ]'))