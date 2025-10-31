from hypothesis import given, assume, strategies as st, settings, HealthCheck
from unittest.mock import Mock
from django.views.generic.edit import ModelFormMixin

# Create a strategy for URLs with placeholders
url_with_placeholder = st.from_regex(r'/[a-z]+/{[a-z_]+}(/[a-z]+)?', fullmatch=True)

@given(success_url_template=url_with_placeholder)
@settings(suppress_health_check=[HealthCheck.filter_too_much], max_examples=10)
def test_modelformmixin_should_not_raise_confusing_keyerror(success_url_template):
    mixin = ModelFormMixin()
    mixin.success_url = success_url_template
    mock_obj = Mock()
    mock_obj.__dict__ = {}  # Empty dict - no attributes
    mixin.object = mock_obj

    try:
        result = mixin.get_success_url()
        print(f"Unexpectedly succeeded for: {success_url_template!r} -> {result}")
    except KeyError as e:
        # This is the problematic behavior
        print(f"KeyError for {success_url_template!r}: {e}")
        return  # We expect this for now
    except Exception as e:
        print(f"Other error for {success_url_template!r}: {type(e).__name__}: {e}")

# Also test manually
print("Manual tests with specific templates:")
print("-" * 50)

test_cases = [
    "/object/{id}/success",
    "/users/{user_id}/profile",
    "/items/{pk}/edit",
    "/foo/{bar}/{baz}",
]

for template in test_cases:
    print(f"\nTesting: {template!r}")
    mixin = ModelFormMixin()
    mixin.success_url = template
    mock_obj = Mock()
    mock_obj.__dict__ = {}
    mixin.object = mock_obj

    try:
        result = mixin.get_success_url()
        print(f"  Result: {result}")
    except KeyError as e:
        print(f"  KeyError: {e}")
    except Exception as e:
        print(f"  Other error: {type(e).__name__}: {e}")

print("\n" + "=" * 50)
print("Running Hypothesis test...")
print("=" * 50)
test_modelformmixin_should_not_raise_confusing_keyerror()