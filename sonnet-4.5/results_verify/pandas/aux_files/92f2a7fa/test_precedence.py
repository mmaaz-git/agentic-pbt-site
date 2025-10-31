#!/usr/bin/env python3
"""Test to understand the operator precedence issue"""

# Simulate the buggy condition
def buggy_condition(s):
    """Current implementation"""
    if not s[0] == "[" and s[-1] == "]":
        return "early return"
    return "continue processing"

# Simulate the fixed condition
def fixed_condition(s):
    """Fixed implementation"""
    if not (s[0] == "[" and s[-1] == "]"):
        return "early return"
    return "continue processing"

test_inputs = [
    "00",           # Neither [ nor ]
    "[1,2,3]",      # Both [ and ]
    "x]",           # Only ends with ]
    "[x",           # Only starts with [
    '{"foo":"bar"}' # JSON object
]

print("Operator precedence analysis:")
print("=" * 60)
print(f"{'Input':<15} {'Buggy':<20} {'Fixed':<20}")
print("-" * 60)

for s in test_inputs:
    buggy = buggy_condition(s)
    fixed = fixed_condition(s)
    print(f"{s:<15} {buggy:<20} {fixed:<20}")

print("\n" + "=" * 60)
print("Breaking down the buggy condition for '00':")
s = "00"
print(f"s = {s!r}")
print(f"s[0] = {s[0]!r}")
print(f"s[-1] = {s[-1]!r}")
print(f"s[0] == '[' = {s[0] == '['}")
print(f"not s[0] == '[' = {not s[0] == '['}")
print(f"s[-1] == ']' = {s[-1] == ']'}")
print(f"(not s[0] == '[') and (s[-1] == ']') = {(not s[0] == '[') and (s[-1] == ']')}")
print(f"Result: {'early return' if (not s[0] == '[') and (s[-1] == ']') else 'continue processing'}")

print("\n" + "=" * 60)
print("Breaking down the fixed condition for '00':")
print(f"s[0] == '[' and s[-1] == ']' = {s[0] == '[' and s[-1] == ']'}")
print(f"not (s[0] == '[' and s[-1] == ']') = {not (s[0] == '[' and s[-1] == ']')}")
print(f"Result: {'early return' if not (s[0] == '[' and s[-1] == ']') else 'continue processing'}")