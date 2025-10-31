from hypothesis import given, strategies as st
from django.conf.urls.static import static
from django.core.exceptions import ImproperlyConfigured
from django.test import override_settings
import pytest
from django.conf import settings

# Configure Django settings
settings.configure(DEBUG=True, SECRET_KEY='test')

@given(st.text(alphabet='/', min_size=1, max_size=10))
@override_settings(DEBUG=True)
def test_static_only_slashes(prefix):
    """Test that static() raises ImproperlyConfigured for slash-only prefixes"""
    print(f"Testing with prefix: '{prefix}'")
    with pytest.raises(ImproperlyConfigured):
        result = static(prefix)
        print(f"  Result: {result}")
        if result:
            print(f"  Pattern: {result[0].pattern.regex.pattern}")

# Run the test
print("Running hypothesis test for slash-only strings...")
try:
    test_static_only_slashes()
    print("Test passed!")
except AssertionError as e:
    print(f"Test failed: {e}")
except Exception as e:
    print(f"Test error: {e}")

# Manual test of specific case
print("\n=== Manual test of '/' ===")
try:
    result = static('/')
    print(f"static('/') succeeded, returned: {result}")
    if result:
        print(f"Pattern: {result[0].pattern.regex.pattern}")
except ImproperlyConfigured as e:
    print(f"static('/') raised ImproperlyConfigured: {e}")