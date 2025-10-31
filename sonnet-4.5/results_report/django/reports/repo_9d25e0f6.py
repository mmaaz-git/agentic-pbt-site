#!/usr/bin/env python3
"""Minimal reproduction of RenameModel immutability bug."""

from django.db.migrations.operations import RenameModel

# Create a RenameModel operation
op = RenameModel("Article", "Post")

# Force the cached properties to be computed before the test
_ = op.old_name_lower
_ = op.new_name_lower

# Print initial state
print(f"Before: old_name={op.old_name!r}, new_name={op.new_name!r}")
print(f"        old_name_lower={op.old_name_lower!r}, new_name_lower={op.new_name_lower!r}")

# Try to call database_backwards with None schema_editor (causes AttributeError)
try:
    op.database_backwards("myapp", None, None, None)
except AttributeError as e:
    print(f"\nException raised: {e}")

# Print state after exception
print(f"\nAfter:  old_name={op.old_name!r}, new_name={op.new_name!r}")
print(f"        old_name_lower={op.old_name_lower!r}, new_name_lower={op.new_name_lower!r}")

# Check if the state was properly restored (it should be unchanged)
if op.old_name == "Article" and op.new_name == "Post":
    print("\n✓ Operation state correctly preserved after exception")
else:
    print("\n✗ BUG: Operation state was mutated after exception!")
    print("  Expected: old_name='Article', new_name='Post'")
    print(f"  Got:      old_name={op.old_name!r}, new_name={op.new_name!r}")