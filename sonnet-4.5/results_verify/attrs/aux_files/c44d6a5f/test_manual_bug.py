#!/usr/bin/env python3
"""Reproduce the bug using the manual example from the bug report."""

from attr.filters import _split_what

items = [int, str, "name1", "name2", float]
gen = (x for x in items)

classes, names, attrs = _split_what(gen)

print(f"Classes: {classes}")
print(f"Names: {names}")
print(f"Attrs: {attrs}")

# Test with a list instead
list_classes, list_names, list_attrs = _split_what(items)
print(f"\nWith list instead of generator:")
print(f"Classes: {list_classes}")
print(f"Names: {list_names}")
print(f"Attrs: {list_attrs}")

# Verify the bug
try:
    assert classes == frozenset({int, str, float})
    print("\n✓ Classes assertion passed")
except AssertionError:
    print("\n✗ Classes assertion failed")

try:
    assert names == frozenset({"name1", "name2"})
    print("✓ Names assertion passed")
except AssertionError:
    print(f"✗ Names assertion failed: expected {frozenset({'name1', 'name2'})}, got {names}")