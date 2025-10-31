#!/usr/bin/env python3

import re

# Test the existing regex pattern from line 351
pattern = r"^(\S*?)([a-zA-Z%!].*)"

test_inputs = [
    "1e-5pt",
    "2.5e3px",
    "1.5pt",
    "100px",
]

print("Testing current regex pattern: r\"^(\\S*?)([a-zA-Z%!].*)\"")
print()

for input_val in test_inputs:
    match = re.match(pattern, input_val)
    if match:
        val, unit = match.groups()
        print(f"Input: {input_val:10} -> val='{val}', unit='{unit}'")
    else:
        print(f"Input: {input_val:10} -> NO MATCH")

print("\n" + "="*50 + "\n")

# Test the proposed fix regex pattern
new_pattern = r"^([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)([a-zA-Z%!].*)"

print("Testing proposed regex pattern:")
print("r\"^([-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?)([a-zA-Z%!].*)\"")
print()

for input_val in test_inputs:
    match = re.match(new_pattern, input_val)
    if match:
        val, unit = match.groups()
        print(f"Input: {input_val:10} -> val='{val}', unit='{unit}'")
    else:
        print(f"Input: {input_val:10} -> NO MATCH")