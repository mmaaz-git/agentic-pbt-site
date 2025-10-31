#!/usr/bin/env python3
"""Test script to verify the typo in attr.assoc docstring"""

import attr

# Get the docstring
docstring = attr.assoc.__doc__

# Check for the typo
if "du to" in docstring:
    print("✓ Typo confirmed: 'du to' found in docstring")
    print("\nContext:")
    for line in docstring.split('\n'):
        if "du to" in line:
            print(f"  '{line.strip()}'")
else:
    print("✗ Typo not found")

# Also check the correct version doesn't exist
if "due to" not in docstring:
    print("✓ Correct version 'due to' is not present")
else:
    print("✗ 'due to' found (should be 'du to' according to bug report)")