#!/usr/bin/env python3
"""
Test reproduction for numpy.f2py.symbolic string concatenation bug
"""

# First, let's try to reproduce the Hypothesis test
from hypothesis import given, strategies as st, settings
from numpy.f2py.symbolic import as_string, normalize, Expr, Op

# Manual reproduction first
print("=== Manual Reproduction ===")
a = as_string('""', 1)
b = as_string('""', 1)
c = as_string("''", 1)

result1 = (a // b) // c
result2 = a // (b // c)

print(f"a = {repr(a)}")
print(f"b = {repr(b)}")
print(f"c = {repr(c)}")
print()
print(f"(a // b) // c = {repr(result1)}")
print(f"normalized: {repr(normalize(result1))}")
print()
print(f"a // (b // c) = {repr(result2)}")
print(f"normalized: {repr(normalize(result2))}")
print()
print(f"Normalized forms equal: {normalize(result1) == normalize(result2)}")
print()

# Now run the hypothesis test
@st.composite
def expr_strings(draw):
    s = draw(st.text(min_size=0, max_size=20))
    quote_char = draw(st.sampled_from(['"', "'"]))
    quoted = quote_char + s + quote_char
    return as_string(quoted, kind=1)

print("=== Running Hypothesis Test ===")
test_passed = True
failure_example = None

@given(expr_strings(), expr_strings(), expr_strings())
@settings(max_examples=500)
def test_string_concat_associative(a, b, c):
    global test_passed, failure_example
    result1 = (a // b) // c
    result2 = a // (b // c)
    if normalize(result1) != normalize(result2):
        test_passed = False
        if failure_example is None:
            failure_example = (a, b, c, result1, result2)

try:
    test_string_concat_associative()
    if test_passed:
        print("Hypothesis test PASSED (no failures found)")
    else:
        print("Hypothesis test FAILED")
        if failure_example:
            a, b, c, r1, r2 = failure_example
            print(f"Failing example:")
            print(f"  a = {repr(a)}")
            print(f"  b = {repr(b)}")
            print(f"  c = {repr(c)}")
            print(f"  normalize((a // b) // c) = {repr(normalize(r1))}")
            print(f"  normalize(a // (b // c)) = {repr(normalize(r2))}")
except Exception as e:
    print(f"Test execution error: {e}")