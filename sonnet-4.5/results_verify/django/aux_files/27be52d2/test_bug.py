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
    id = 42

mixin2.object = MockObject2()

try:
    result = mixin2.get_success_url()
    print(f"SUCCESS: Result = {result}")
except Exception as e:
    print(f"Error: {e}")

# Test 3: Property-based test scenario from report
print("\nTest 3: Running property-based test scenario...")
from hypothesis import given, settings as hypo_settings, strategies as st, assume

@given(
    placeholder_key=st.text(alphabet='abcdefghijklmnopqrstuvwxyz_', min_size=1, max_size=10),
    object_attrs=st.dictionaries(
        st.text(alphabet='abcdefghijklmnopqrstuvwxyz_', min_size=1, max_size=10),
        st.text(max_size=20),
        max_size=5
    )
)
@hypo_settings(max_examples=10)
def test_deletionmixin_success_url_formatting(placeholder_key, object_attrs):
    assume(placeholder_key not in object_attrs)

    class TestMixin(DeletionMixin):
        def __init__(self):
            super().__init__()
            self.success_url = f'/path/{{{placeholder_key}}}/'

    mixin = TestMixin()

    class MockObject:
        def __init__(self, attrs):
            self.__dict__.update(attrs)

    mixin.object = MockObject(object_attrs)

    try:
        result = mixin.get_success_url()
        return False  # Should have raised an error
    except KeyError:
        return True  # Expected behavior
    except Exception:
        return False  # Unexpected error

# Run a few examples
print("Running property-based tests...")
test_deletionmixin_success_url_formatting()
print("Property-based tests completed.")

# Test 4: Check the same pattern in ModelFormMixin
print("\nTest 4: Checking ModelFormMixin for consistency...")
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
except Exception as e:
    print(f"Unexpected error: {e}")

print("\nAll tests completed.")