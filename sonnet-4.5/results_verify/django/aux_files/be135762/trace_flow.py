#!/usr/bin/env python3
"""Trace the flow of Variable creation for '2.'"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Let's trace what happens step by step
var = "2."

print(f"Input: '{var}'")
print()

# Step 1: Check if it contains "." or "e"
if "." in var or "e" in var.lower():
    print("Step 1: Contains '.', so going to try float()")

    # Step 2: Convert to float
    try:
        literal = float(var)
        print(f"Step 2: float('{var}') = {literal}")

        # Step 3: Check if last character is "."
        if var[-1] == ".":
            print("Step 3: Last character is '.', raising ValueError")
            raise ValueError("Trailing dot not allowed")
        print("Step 3: Last character is not '.', would accept the float")
    except ValueError as e:
        print(f"Step 4: ValueError caught: {e}")
        print("This ValueError gets caught by the outer except block")
        print("The code then treats '2.' as a variable name and splits on '.'")

        # This is what happens in the actual code
        VARIABLE_ATTRIBUTE_SEPARATOR = "."
        lookups = tuple(var.split(VARIABLE_ATTRIBUTE_SEPARATOR))
        print(f"Result: lookups = {lookups}")
        print("Note: lookups contains an empty string as the second element!")

print()
print("The bug is that the ValueError raised to reject '2.' is caught")
print("by the same except block that handles non-numeric variables.")