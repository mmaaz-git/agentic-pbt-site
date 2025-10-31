#!/usr/bin/env python3
"""
Test to verify r_empty regex behavior with empty string
"""

import re

r_empty = re.compile(r'^\s+$')

# Test various inputs
test_cases = [
    "",            # Empty string
    " ",           # Single space
    "  ",          # Two spaces
    "\t",          # Tab
    "\n",          # Newline (shouldn't match because $ matches before \n)
    " \n",         # Space + newline
]

print("Testing r_empty regex pattern r'^\\s+$':")
for test in test_cases:
    matches = r_empty.match(test)
    print(f"'{repr(test)}' -> {'matches' if matches else 'does NOT match'}")