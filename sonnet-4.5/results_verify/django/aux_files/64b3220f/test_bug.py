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

from django.views.generic.edit import ModelFormMixin
from django.contrib.auth.models import User

print("Test 1: Basic reproduction from bug report")
print("-" * 50)

class TestMixin(ModelFormMixin):
    def __init__(self):
        super().__init__()
        self.success_url = '/user/{user_id}/detail/'
        self.model = User

mixin = TestMixin()

class MockObject:
    pass

mixin.object = MockObject()

try:
    result = mixin.get_success_url()
    print(f"Result: {result}")
except KeyError as e:
    print(f"KeyError raised: {e}")
except Exception as e:
    print(f"Other exception: {type(e).__name__}: {e}")

print("\nTest 2: Hypothesis test case")
print("-" * 50)

placeholder_key = '_'
object_attrs = {}

class TestMixin2(ModelFormMixin):
    def __init__(self):
        super().__init__()
        self.success_url = f'/path/{{{placeholder_key}}}/'
        self.model = User

mixin2 = TestMixin2()

class MockObject2:
    def __init__(self, attrs):
        self.__dict__.update(attrs)

mixin2.object = MockObject2(object_attrs)

try:
    result = mixin2.get_success_url()
    print(f"Result: {result}")
except KeyError as e:
    print(f"KeyError raised: {e}")
except Exception as e:
    print(f"Other exception: {type(e).__name__}: {e}")

print("\nTest 3: Valid case with matching attributes")
print("-" * 50)

class TestMixin3(ModelFormMixin):
    def __init__(self):
        super().__init__()
        self.success_url = '/user/{user_id}/detail/'
        self.model = User

mixin3 = TestMixin3()

class MockObject3:
    def __init__(self):
        self.user_id = 123

mixin3.object = MockObject3()

try:
    result = mixin3.get_success_url()
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")

print("\nTest 4: Check what happens with no success_url but no get_absolute_url")
print("-" * 50)

class TestMixin4(ModelFormMixin):
    def __init__(self):
        super().__init__()
        self.success_url = None
        self.model = User

mixin4 = TestMixin4()
mixin4.object = MockObject()

try:
    result = mixin4.get_success_url()
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")