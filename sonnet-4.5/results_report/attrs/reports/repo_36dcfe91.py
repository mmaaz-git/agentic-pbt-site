from attr.filters import _split_what

# Test case demonstrating the bug
items = [int, str, "name1", "name2", float]

# Test with generator (bug occurs here)
gen = (x for x in items)
classes_gen, names_gen, attrs_gen = _split_what(gen)

print("=== Generator Input ===")
print(f"Classes: {classes_gen}")
print(f"Names: {names_gen}")
print(f"Attrs: {attrs_gen}")

# Test with list (correct behavior)
classes_list, names_list, attrs_list = _split_what(items)

print("\n=== List Input ===")
print(f"Classes: {classes_list}")
print(f"Names: {names_list}")
print(f"Attrs: {attrs_list}")

# Verify the bug
print("\n=== Assertion Checks ===")
try:
    assert classes_gen == frozenset({int, str, float})
    print("✓ Classes assertion passed")
except AssertionError:
    print("✗ Classes assertion failed")

try:
    assert names_gen == frozenset({"name1", "name2"})
    print("✓ Names assertion passed")
except AssertionError:
    print("✗ Names assertion failed - Expected {'name1', 'name2'}, got:", names_gen)

try:
    assert attrs_gen == frozenset()
    print("✓ Attrs assertion passed (no Attribute objects)")
except AssertionError:
    print("✗ Attrs assertion failed")