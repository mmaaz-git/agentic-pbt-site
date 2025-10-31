#!/usr/bin/env python3
"""
Minimal reproduction of Django TemplateCommand camel case conversion bug
with digits in app/project names.
"""

def to_camel_case(name):
    """From Django's management/templates.py line 138"""
    return "".join(x for x in name.title() if x != "_")


# Main test case
print("=== Django Camel Case Conversion Bug ===")
print()
print("Input: 'my2app'")
print(f"Step 1 - .title(): {'my2app'.title()}")
print(f"Step 2 - remove underscores: {to_camel_case('my2app')}")
print(f"Expected: 'My2app' (digit doesn't create word boundary)")
print(f"Actual: 'My2App' (letter after digit is capitalized)")
print()
print("ERROR: The 'a' after '2' is incorrectly capitalized")
print()

# Additional examples
print("=== Additional Examples ===")
examples = [
    ("test1module", "Test1module", "Test1Module"),
    ("app2api", "App2api", "App2Api"),
    ("my3rd_app", "My3rdApp", "My3RdApp"),
    ("version2_0", "Version20", "Version20"),  # This one works as expected
    ("my_2_app", "My2App", "My2App"),  # This one works as expected
]

print("Input         -> Expected     -> Actual       | Correct?")
print("-" * 55)
for input_name, expected, actual in examples:
    result = to_camel_case(input_name)
    is_correct = result == expected
    print(f"{input_name:13} -> {expected:12} -> {result:12} | {'✓' if is_correct else '✗'}")
    if result != actual:
        print(f"  WARNING: Got '{result}' but example shows '{actual}'")

print()
print("=== Analysis ===")
print("Python's .title() method treats digits as word boundaries.")
print("This causes unexpected capitalization for valid Python identifiers.")
print("App names with digits (e.g., 'my2app') are valid but produce")
print("counter-intuitive camel case class names (e.g., 'My2AppConfig').")