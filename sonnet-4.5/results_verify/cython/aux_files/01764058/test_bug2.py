#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Utils import build_hex_version

# Test to verify the error message issue
def check_error_message(version_str):
    """Check if the error message contains 'invalid literal for int()'"""
    try:
        result = build_hex_version(version_str)
        return None  # No error
    except ValueError as e:
        if "invalid literal for int()" in str(e):
            return True  # Unhelpful error message
        else:
            return False  # Helpful error message
    except Exception as e:
        return None  # Other error

print("Checking for unhelpful error messages:")
print("=" * 50)

test_cases = [
    ("", "empty string"),
    (".", "just a dot"),
    ("a", "alphabetic character"),
    ("1.0foo", "version with invalid suffix"),
    ("1..0", "double dot"),
]

for test_input, description in test_cases:
    result = check_error_message(test_input)
    if result is True:
        print(f"✗ {description:30} - Has unhelpful 'invalid literal for int()' error")
    elif result is False:
        print(f"✓ {description:30} - Has helpful error message")
    else:
        print(f"? {description:30} - Different or no error")