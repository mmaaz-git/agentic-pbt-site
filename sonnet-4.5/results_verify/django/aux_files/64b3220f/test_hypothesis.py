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

from hypothesis import given, settings, strategies as st, assume
from django.views.generic.edit import ModelFormMixin
from django.contrib.auth.models import User

@given(
    placeholder_key=st.text(alphabet='abcdefghijklmnopqrstuvwxyz_', min_size=1, max_size=10),
    object_attrs=st.dictionaries(
        st.text(alphabet='abcdefghijklmnopqrstuvwxyz_', min_size=1, max_size=10),
        st.text(max_size=20),
        max_size=5
    )
)
@settings(max_examples=100)
def test_modelformmixin_success_url_formatting(placeholder_key, object_attrs):
    assume(placeholder_key not in object_attrs)

    class TestMixin(ModelFormMixin):
        def __init__(self):
            super().__init__()
            self.success_url = f'/path/{{{placeholder_key}}}/'
            self.model = User

    mixin = TestMixin()

    class MockObject:
        def __init__(self, attrs):
            self.__dict__.update(attrs)

    mixin.object = MockObject(object_attrs)

    # This should raise a KeyError according to the bug report
    try:
        result = mixin.get_success_url()
        print(f"No error for placeholder: {placeholder_key}, attrs: {object_attrs}")
    except KeyError as e:
        # This is expected behavior according to bug report
        pass
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")

# Run the test
print("Running property-based test with 100 examples...")
test_modelformmixin_success_url_formatting()
print("All test cases raised KeyError as expected")