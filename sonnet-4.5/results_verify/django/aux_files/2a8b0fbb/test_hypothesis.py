import django
from django.conf import settings
from django.apps import apps
from hypothesis import given, settings as hyp_settings, strategies as st
import pytest

# Configure Django
if not settings.configured:
    settings.configure(
        INSTALLED_APPS=['django.contrib.contenttypes'],
        DATABASES={'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': ':memory:'}},
        SECRET_KEY='test',
    )
django.setup()

# Property-based test from the bug report
@given(st.text(min_size=1, max_size=50).filter(lambda x: '.' not in x))
@hyp_settings(max_examples=10)
def test_get_model_no_dot_clear_error(app_label_no_dot):
    """Test that get_model raises ValueError with clear message for strings without dots."""
    with pytest.raises(ValueError) as exc_info:
        apps.get_model(app_label_no_dot)

    error_msg = str(exc_info.value)
    # Check if error message is clear about the format requirement
    is_clear = "exactly one dot" in error_msg or "format" in error_msg

    if not is_clear:
        print(f"Unclear error for input '{app_label_no_dot}': {error_msg}")
        return False
    return True

# Run the test manually with specific examples
test_inputs = ["contenttypes", "myapp", "a", "test_app_name"]

print("Testing specific inputs:")
print("=" * 50)

for test_input in test_inputs:
    print(f"\nInput: '{test_input}'")
    try:
        apps.get_model(test_input)
    except ValueError as e:
        error_msg = str(e)
        is_clear = "exactly one dot" in error_msg or "format" in error_msg
        print(f"Error message: {error_msg}")
        print(f"Is clear about format requirement: {is_clear}")
        if not is_clear:
            print("❌ FAIL: Error message is not clear about the format requirement")