def to_camel_case(name):
    """From Django's management/templates.py line 138"""
    return "".join(x for x in name.title() if x != "_")


print("Input: 'my2app'")
print(f"Step 1 - .title(): {'my2app'.title()}")
print(f"Step 2 - remove underscores: {to_camel_case('my2app')}")
print(f"Expected: 'My2app'")
print(f"Actual: 'My2App'")
print()
print("The 'a' after '2' is incorrectly capitalized")

print("\nMore examples:")
examples = ["test1module", "app2api", "my_2_app"]
for ex in examples:
    result = to_camel_case(ex)
    print(f"  '{ex}' -> '{result}'")