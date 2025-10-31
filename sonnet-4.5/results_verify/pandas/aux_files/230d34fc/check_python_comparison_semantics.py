#!/usr/bin/env python3
"""Check Python's standard comparison semantics."""

# Test standard Python objects to understand expected behavior
print("Testing standard Python comparison semantics:")
print("=" * 50)

# Test with numbers
print("With numbers:")
x = 5
print(f"5 == 5: {x == x}")
print(f"5 <= 5: {x <= x}")
print(f"5 >= 5: {x >= x}")
print("Consistent: Yes (all True)")

# Test with strings
print("\nWith strings:")
s = "hello"
print(f"'hello' == 'hello': {s == s}")
print(f"'hello' <= 'hello': {s <= s}")
print(f"'hello' >= 'hello': {s >= s}")
print("Consistent: Yes (all True)")

# Test with float infinity
print("\nWith float('inf'):")
import math
inf = float('inf')
print(f"inf == inf: {inf == inf}")
print(f"inf <= inf: {inf <= inf}")
print(f"inf >= inf: {inf >= inf}")
print("Consistent: Yes (all True)")

# Test with float negative infinity
print("\nWith float('-inf'):")
neginf = float('-inf')
print(f"-inf == -inf: {neginf == neginf}")
print(f"-inf <= -inf: {neginf <= neginf}")
print(f"-inf >= -inf: {neginf >= neginf}")
print("Consistent: Yes (all True)")

print("\n" + "=" * 50)
print("CONCLUSION:")
print("In Python, for any value x where x == x is True,")
print("both x <= x and x >= x must also be True.")
print("This is a fundamental property of comparison operators.")
print("\nThe <= operator means 'less than OR equal to'.")
print("If x == x, then x is equal to x, so x <= x must be True.")
print("Similarly for >= ('greater than OR equal to').")