#!/usr/bin/env python3
"""Test the DeletionMixin bug report"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

import django
from django.conf import settings
if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY='test-secret-key',
        DATABASES={'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': ':memory:'}},
        INSTALLED_APPS=['django.contrib.contenttypes', 'django.contrib.auth'],
        USE_TZ=True,
    )
    django.setup()

from django.views.generic.edit import DeletionMixin

# Test 1: Reproduce the exact bug
print("Test 1: Reproducing the bug with missing placeholder...")
class TestMixin(DeletionMixin):
    def __init__(self):
        super().__init__()
        self.success_url = '/category/{category_id}/'

mixin = TestMixin()

class MockObject:
    id = 42

mixin.object = MockObject()

try:
    result = mixin.get_success_url()
    print(f"SUCCESS: Result = {result}")
except KeyError as e:
    print(f"KeyError raised as expected: {e}")
    print(f"  This is a bare KeyError with message: '{e}'")
except Exception as e:
    print(f"Unexpected error: {e}")

# Test 2: Test when the placeholder exists
print("\nTest 2: Testing with matching placeholder...")
class TestMixin2(DeletionMixin):
    def __init__(self):
        super().__init__()
        self.success_url = '/object/{id}/'

mixin2 = TestMixin2()

class MockObject2:
    def __init__(self):
        self.id = 42  # Setting as attribute via __init__

mixin2.object = MockObject2()

try:
    result = mixin2.get_success_url()
    print(f"SUCCESS: Result = {result}")
except Exception as e:
    print(f"Error: {e}")

# Test 3: Check the same pattern in ModelFormMixin
print("\nTest 3: Checking ModelFormMixin for consistency...")
from django.views.generic.edit import ModelFormMixin

class TestModelFormMixin(ModelFormMixin):
    def __init__(self):
        super().__init__()
        self.success_url = '/category/{category_id}/'

mixin3 = TestModelFormMixin()
mixin3.object = MockObject()

try:
    result = mixin3.get_success_url()
    print(f"SUCCESS: Result = {result}")
except KeyError as e:
    print(f"ModelFormMixin also raises KeyError: {e}")
    print(f"  This is a bare KeyError with message: '{e}'")
except Exception as e:
    print(f"Unexpected error: {e}")

# Test 4: Verify line numbers
print("\nTest 4: Checking source code lines...")
import inspect
source = inspect.getsource(DeletionMixin.get_success_url)
lines = source.split('\n')
for i, line in enumerate(lines, start=234):  # Starting from the actual line number in the file
    print(f"Line {i}: {line}")

print("\nAll tests completed.")