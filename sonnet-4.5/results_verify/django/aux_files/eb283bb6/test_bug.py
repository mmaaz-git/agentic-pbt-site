#!/usr/bin/env python
import sys
import os
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.template import Variable, Context
from hypothesis import given, settings, strategies as st, example

print("Testing the bug report claim about '42.' handling in django.template.Variable\n")
print("=" * 60)

# First, let's test the basic reproduction case
print("\n1. Basic reproduction test:")
print("-" * 30)
try:
    var = Variable('42.')
    print(f"Variable created successfully for '42.'")
    print(f"  literal: {var.literal}")
    print(f"  lookups: {var.lookups}")

    ctx = Context({})
    try:
        result = var.resolve(ctx)
        print(f"  resolve() result: {result}")
    except Exception as e:
        print(f"  resolve() raised: {type(e).__name__}: {e}")
except Exception as e:
    print(f"Failed to create Variable: {type(e).__name__}: {e}")

# Test other similar cases
print("\n2. Testing other numeric strings:")
print("-" * 30)
test_cases = ['42', '42.0', '42.', '0.', '123.', '.42', '42e2', '42.e2']
for test_str in test_cases:
    try:
        var = Variable(test_str)
        print(f"\n'{test_str}':")
        print(f"  literal: {var.literal}")
        print(f"  lookups: {var.lookups}")
        ctx = Context({})
        try:
            result = var.resolve(ctx)
            print(f"  resolve() result: {result}")
        except Exception as e:
            print(f"  resolve() raised: {type(e).__name__}")
    except Exception as e:
        print(f"\n'{test_str}': Failed to create Variable - {type(e).__name__}")

# Now run the hypothesis test
print("\n3. Running Hypothesis test:")
print("-" * 30)

@settings(max_examples=10, database=None)
@example(n=42)
@example(n=0)
@example(n=123)
@given(st.integers())
def test_variable_numeric_string_with_trailing_dot_should_be_resolvable(n):
    s = str(n) + '.'
    var = Variable(s)

    if var.literal is not None:
        ctx = Context({})
        try:
            resolved = var.resolve(ctx)
            assert resolved == var.literal, f"Expected {var.literal}, got {resolved}"
            return True
        except Exception as e:
            print(f"  Failed for n={n} ('{s}'): resolve() raised {type(e).__name__}")
            return False
    return True

try:
    test_variable_numeric_string_with_trailing_dot_should_be_resolvable()
    print("Hypothesis test completed")
except AssertionError as e:
    print(f"Hypothesis test assertion failed: {e}")
except Exception as e:
    print(f"Hypothesis test failed with: {type(e).__name__}: {e}")

# Let's also trace through the exact behavior described in the bug report
print("\n4. Detailed trace of '42.' processing:")
print("-" * 30)

class TracedVariable(Variable):
    def __init__(self, var):
        self.var = var
        self.literal = None
        self.lookups = None
        self.translate = False
        self.message_context = None

        if not isinstance(var, str):
            raise TypeError("Variable must be a string or number, got %s" % type(var))

        print(f"  Processing '{var}':")
        try:
            # First try to treat this variable as a number.
            if "." in var or "e" in var.lower():
                print(f"    Attempting float('{var}')...")
                self.literal = float(var)
                print(f"    Success! literal = {self.literal}")
                # "2." is invalid
                if var[-1] == ".":
                    print(f"    But var ends with '.', raising ValueError")
                    raise ValueError
            else:
                self.literal = int(var)
        except ValueError as e:
            print(f"    ValueError caught, literal is currently: {self.literal}")
            print(f"    Now setting lookups...")
            # Skip translation and quote handling for simplicity
            self.lookups = tuple(var.split('.'))
            print(f"    lookups = {self.lookups}")

traced_var = TracedVariable('42.')
print(f"\nFinal state:")
print(f"  literal: {traced_var.literal}")
print(f"  lookups: {traced_var.lookups}")