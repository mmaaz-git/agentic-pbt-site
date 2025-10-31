from unittest.mock import Mock
from django.views.generic.edit import ModelFormMixin

# First test: Reproduce the exact bug
print("Test 1: Reproducing the exact bug described")
print("-" * 50)

mixin = ModelFormMixin()
mixin.success_url = "/object/{id}/success"
mock_obj = Mock()
mock_obj.__dict__ = {}
mixin.object = mock_obj

try:
    result = mixin.get_success_url()
    print(f"Unexpected success: {result}")
except KeyError as e:
    print(f"Got KeyError as described: {e}")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
except Exception as e:
    print(f"Got different exception: {type(e).__name__}: {e}")

print("\n")

# Test 2: Test with a valid object
print("Test 2: Testing with valid object attributes")
print("-" * 50)

mixin2 = ModelFormMixin()
mixin2.success_url = "/object/{id}/success"
mock_obj2 = Mock()
mock_obj2.__dict__ = {'id': 123}
mixin2.object = mock_obj2

try:
    result2 = mixin2.get_success_url()
    print(f"Success with valid object: {result2}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")

print("\n")

# Test 3: Test with multiple placeholders
print("Test 3: Testing with multiple placeholders")
print("-" * 50)

mixin3 = ModelFormMixin()
mixin3.success_url = "/object/{id}/{name}/success"
mock_obj3 = Mock()
mock_obj3.__dict__ = {'id': 123}  # Missing 'name'
mixin3.object = mock_obj3

try:
    result3 = mixin3.get_success_url()
    print(f"Unexpected success: {result3}")
except KeyError as e:
    print(f"Got KeyError for missing 'name': {e}")
except Exception as e:
    print(f"Got different exception: {type(e).__name__}: {e}")