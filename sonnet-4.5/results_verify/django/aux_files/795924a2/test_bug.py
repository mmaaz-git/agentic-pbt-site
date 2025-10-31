#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.template import Variable, Context

print("Testing the property-based test case...")

# Manually test the logic from the property test
def test_variable_literal_trailing_dot_manual(num, add_trailing_dot):
    var_string = str(num)
    if add_trailing_dot:
        var_string = var_string + '.'

    try:
        v = Variable(var_string)

        if v.literal is not None and isinstance(v.literal, float):
            assert var_string[-1] != '.', f"Variable with trailing dot '{var_string}' should not parse as valid float but got {v.literal}"
    except ValueError:
        pass

# Test the specific failing case
print("\nTesting specific failing case: num=0, add_trailing_dot=True")
try:
    test_variable_literal_trailing_dot_manual(0, True)
    print("Property test passed (no assertion error)")
except AssertionError as e:
    print(f"Property test FAILED: {e}")
except Exception as e:
    print(f"Property test error: {e}")

# Test the basic reproduction case
print("\n\nTesting the basic reproduction case with '2.':")
print("=" * 50)

v = Variable("2.")

print(f"v.var = {v.var}")
print(f"v.literal = {v.literal}")
print(f"v.lookups = {v.lookups}")

ctx = Context({"2": {"": "unexpected_value"}})
result = v.resolve(ctx)
print(f"\nContext has key '2': {'2' in ctx}")
print(f"Result when resolving '2.' = {result}")

# Test with a normal float
print("\n\nTesting normal float '2.0':")
print("=" * 50)
v2 = Variable("2.0")
print(f"v2.var = {v2.var}")
print(f"v2.literal = {v2.literal}")
print(f"v2.lookups = {v2.lookups}")
result2 = v2.resolve(ctx)
print(f"Result when resolving '2.0' = {result2}")

# Test with just an integer
print("\n\nTesting integer '2':")
print("=" * 50)
v3 = Variable("2")
print(f"v3.var = {v3.var}")
print(f"v3.literal = {v3.literal}")
print(f"v3.lookups = {v3.lookups}")
result3 = v3.resolve(ctx)
print(f"Result when resolving '2' = {result3}")

# Test with other trailing dot cases
print("\n\nTesting more trailing dot cases:")
print("=" * 50)

test_cases = ["0.", "123.", "42.", "-5."]
for case in test_cases:
    print(f"\nCase: '{case}'")
    try:
        v = Variable(case)
        print(f"  literal = {v.literal}")
        print(f"  lookups = {v.lookups}")

        # Check the invariant mentioned in the bug report
        if v.literal is not None and v.lookups is not None:
            print(f"  WARNING: Both literal AND lookups are set!")
            print(f"  This violates the expected invariant")

    except Exception as e:
        print(f"  Exception: {e}")