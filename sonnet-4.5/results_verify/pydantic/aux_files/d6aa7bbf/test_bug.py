from hypothesis import given, strategies as st
from pydantic.alias_generators import to_pascal, to_snake

# First test the hypothesis test
snake_case_strategy = st.from_regex(r'^[a-z]+(_[a-z]+)*$', fullmatch=True)

@given(snake_case_strategy)
def test_snake_with_numbers_roundtrip(s):
    s_with_num = s + '1' if s else 'a1'

    pascal = to_pascal(s_with_num)
    back = to_snake(pascal)

    assert back == s_with_num, \
        f"Round-trip with number failed: {s_with_num} -> {pascal} -> {back}"

# Try to run it on the specific failing case
print("Testing hypothesis test on specific case:")
try:
    s = 'aa'
    s_with_num = s + '1' if s else 'a1'
    pascal = to_pascal(s_with_num)
    back = to_snake(pascal)
    assert back == s_with_num, f"Round-trip with number failed: {s_with_num} -> {pascal} -> {back}"
    print("Hypothesis test passed (unexpected)")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")

# Now reproduce the specific example
print("\n" + "="*50)
print("Reproducing the specific bug examples:")
print("="*50)

from pydantic.alias_generators import to_pascal, to_snake

# Test case 1: aa1
original = 'aa1'
pascal = to_pascal(original)
result = to_snake(pascal)
print(f"\nTest 1: {original} -> {pascal} -> {result}")
print(f"  Expected: {original}")
print(f"  Got: {result}")
print(f"  Match: {result == original}")

# Test case 2: field1
original = 'field1'
pascal = to_pascal(original)
result = to_snake(pascal)
print(f"\nTest 2: {original} -> {pascal} -> {result}")
print(f"  Expected: {original}")
print(f"  Got: {result}")
print(f"  Match: {result == original}")

# Test case 3: test2
original = 'test2'
pascal = to_pascal(original)
result = to_snake(pascal)
print(f"\nTest 3: {original} -> {pascal} -> {result}")
print(f"  Expected: {original}")
print(f"  Got: {result}")
print(f"  Match: {result == original}")

# Test case 4: var3x
original = 'var3x'
pascal = to_pascal(original)
result = to_snake(pascal)
print(f"\nTest 4: {original} -> {pascal} -> {result}")
print(f"  Expected: {original}")
print(f"  Got: {result}")
print(f"  Match: {result == original}")

# Test case 5: hash256
original = 'hash256'
pascal = to_pascal(original)
result = to_snake(pascal)
print(f"\nTest 5: {original} -> {pascal} -> {result}")
print(f"  Expected: {original}")
print(f"  Got: {result}")
print(f"  Match: {result == original}")

# Let's also test some cases without numbers to see if round-trip works there
print("\n" + "="*50)
print("Testing round-trip without numbers:")
print("="*50)

test_cases = ['hello', 'hello_world', 'my_variable_name', 'snake_case_example']
for original in test_cases:
    pascal = to_pascal(original)
    result = to_snake(pascal)
    print(f"\n{original} -> {pascal} -> {result}")
    print(f"  Match: {result == original}")

# Let's check what happens with already snake_case with underscores before numbers
print("\n" + "="*50)
print("Testing existing snake_case with underscores before numbers:")
print("="*50)

test_cases = ['field_1', 'test_2', 'var_3_x', 'hash_256']
for original in test_cases:
    pascal = to_pascal(original)
    result = to_snake(pascal)
    print(f"\n{original} -> {pascal} -> {result}")
    print(f"  Match: {result == original}")