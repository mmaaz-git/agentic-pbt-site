#!/usr/bin/env python3
"""
Minimal reproduction case for AddIndex.reduce() mutation bug in Django
"""
from django.db.migrations.operations import AddIndex, RenameIndex
from django.db import models

# Create an index with an initial name
index = models.Index(fields=['name'], name='old_idx')

# Create an AddIndex operation
add_op = AddIndex(model_name='TestModel', index=index)

# Print the initial state
print(f'Before reduce: {add_op.index.name}')

# Create a RenameIndex operation
rename_op = RenameIndex(model_name='TestModel', old_name='old_idx', new_name='new_idx')

# Call reduce() which should NOT mutate the original operation
result = add_op.reduce(rename_op, 'test_app')

# Check the state after reduce
print(f'After reduce: {add_op.index.name}')
print(f'Expected: old_idx')
print(f'Actual: {add_op.index.name}')

# Show that the mutation happened
if add_op.index.name != 'old_idx':
    print('\n❌ BUG CONFIRMED: The original AddIndex operation was mutated!')
    print('   The index.name changed from "old_idx" to "new_idx"')
else:
    print('\n✓ No bug: The original operation was not mutated')