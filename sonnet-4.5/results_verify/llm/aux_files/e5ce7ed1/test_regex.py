import re

# Test Python regex behavior with $ and \Z
pattern_dollar = re.compile(r"^[\w-]+$")
pattern_Z = re.compile(r"^[\w-]+\Z")

test_cases = [
    "utf8_general_ci",      # Valid
    "utf8_general_ci\n",    # Has trailing newline
    "utf8-general-ci",      # Valid with hyphen
    "utf8_general_ci\n\n",  # Multiple newlines
    "\nutf8_general_ci",    # Leading newline
    "utf8\ngeneral_ci",     # Embedded newline
]

print("Pattern with $ (current behavior):")
for test in test_cases:
    match = pattern_dollar.match(test)
    print(f"  {repr(test):25} -> {'MATCH' if match else 'NO MATCH'}")

print("\nPattern with \\Z (proposed fix):")
for test in test_cases:
    match = pattern_Z.match(test)
    print(f"  {repr(test):25} -> {'MATCH' if match else 'NO MATCH'}")