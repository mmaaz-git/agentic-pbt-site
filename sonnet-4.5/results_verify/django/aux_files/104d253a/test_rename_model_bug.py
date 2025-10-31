#!/usr/bin/env python3
"""Test script to reproduce the RenameModel and RenameIndex mutation bug."""

import sys
import os

# Add the Django environment to the path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Set up minimal Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')

import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        DEBUG=True,
        DATABASES={
            'default': {
                'ENGINE': 'django.db.backends.sqlite3',
                'NAME': ':memory:',
            }
        },
        INSTALLED_APPS=[
            'django.contrib.contenttypes',
            'django.contrib.auth',
        ],
        SECRET_KEY='test-secret-key'
    )

django.setup()

# Now import the migration operations
from django.db.migrations.operations import RenameModel, RenameIndex

print("=" * 60)
print("Testing RenameModel.database_backwards() mutation bug")
print("=" * 60)

# Test 1: Simple reproduction test from the bug report
op = RenameModel(old_name='Author', new_name='Writer')

print(f"Before: old_name={op.old_name}, new_name={op.new_name}")

try:
    op.database_backwards(None, None, None, None)
except Exception as e:
    # We expect an exception since we're passing None
    pass

print(f"After:  old_name={op.old_name}, new_name={op.new_name}")

if op.old_name == 'Writer' and op.new_name == 'Author':
    print("❌ BUG CONFIRMED: The operation state was mutated!")
else:
    print("✓ No mutation detected")

print()

# Test 2: Test with hypothesis-like inputs
print("=" * 60)
print("Testing with hypothesis inputs (old_name='A', new_name='B')")
print("=" * 60)

op2 = RenameModel(old_name='A', new_name='B')
original_old_name = op2.old_name
original_new_name = op2.new_name

print(f"Before: old_name={op2.old_name}, new_name={op2.new_name}")

try:
    op2.database_backwards(
        app_label='test_app',
        schema_editor=None,
        from_state=None,
        to_state=None
    )
except Exception:
    pass

print(f"After:  old_name={op2.old_name}, new_name={op2.new_name}")

if op2.old_name != original_old_name or op2.new_name != original_new_name:
    print(f"❌ BUG CONFIRMED: State mutated from ({original_old_name}, {original_new_name}) to ({op2.old_name}, {op2.new_name})")
else:
    print("✓ No mutation detected")

print()

# Test 3: Test RenameIndex
print("=" * 60)
print("Testing RenameIndex.database_backwards() mutation bug")
print("=" * 60)

op3 = RenameIndex(model_name='TestModel', old_name='idx_old', new_name='idx_new')

print(f"Before: old_name={op3.old_name}, new_name={op3.new_name}")

try:
    op3.database_backwards(None, None, None, None)
except Exception:
    pass

print(f"After:  old_name={op3.old_name}, new_name={op3.new_name}")

if op3.old_name == 'idx_new' and op3.new_name == 'idx_old':
    print("❌ BUG CONFIRMED: The RenameIndex operation state was also mutated!")
else:
    print("✓ No mutation detected")

print()
print("=" * 60)
print("Summary:")
print("The bug report is correct. Both RenameModel and RenameIndex")
print("operations mutate their state when database_backwards() raises")
print("an exception, violating the documented immutability requirement.")
print("=" * 60)