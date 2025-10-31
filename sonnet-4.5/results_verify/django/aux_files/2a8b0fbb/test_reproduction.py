import django
from django.conf import settings
from django.apps import apps

# Configure Django
if not settings.configured:
    settings.configure(
        INSTALLED_APPS=['django.contrib.contenttypes'],
        DATABASES={'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': ':memory:'}},
        SECRET_KEY='test',
    )
django.setup()

print("Testing apps.get_model() with invalid inputs:")
print("=" * 50)

# Test 1: String without a dot
print("\nTest 1: String without a dot ('contenttypes')")
try:
    apps.get_model("contenttypes")
except ValueError as e:
    print(f"Caught ValueError: {e}")
except Exception as e:
    print(f"Caught unexpected exception: {type(e).__name__}: {e}")

# Test 2: String with multiple dots
print("\nTest 2: String with multiple dots ('content.types.model')")
try:
    apps.get_model("content.types.model")
except ValueError as e:
    print(f"Caught ValueError: {e}")
except Exception as e:
    print(f"Caught unexpected exception: {type(e).__name__}: {e}")

# Test 3: Valid usage for comparison
print("\nTest 3: Valid usage ('contenttypes.ContentType')")
try:
    model = apps.get_model("contenttypes.ContentType")
    print(f"Success: Got model {model}")
except Exception as e:
    print(f"Caught exception: {type(e).__name__}: {e}")