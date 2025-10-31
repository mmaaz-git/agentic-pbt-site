from hypothesis import given, assume, strategies as st
from unittest.mock import Mock
from django.views.generic.edit import ModelFormMixin


@given(success_url_template=st.text(min_size=1, max_size=100))
def test_modelformmixin_should_not_raise_confusing_keyerror(success_url_template):
    assume('{' in success_url_template and '}' in success_url_template)

    mixin = ModelFormMixin()
    mixin.success_url = success_url_template
    mock_obj = Mock()
    mock_obj.__dict__ = {}
    mixin.object = mock_obj

    try:
        result = mixin.get_success_url()
    except KeyError as e:
        # This is what the bug report considers problematic
        print(f"KeyError raised for template: {success_url_template!r}")
        print(f"KeyError message: {e}")
        raise AssertionError(
            f"get_success_url() should not raise KeyError for URL {success_url_template!r}. "
            "It should either validate the template or provide a helpful error message."
        )

# Run the test with a few examples
print("Running Hypothesis test with specific examples:")
print("-" * 50)

test_cases = [
    "/object/{id}/success",
    "/users/{user_id}/profile",
    "/items/{pk}/edit",
    "/foo/{bar}/{baz}",
    "/{0}/test"  # This one is interesting - numeric placeholder
]

for template in test_cases:
    if '{' in template and '}' in template:
        print(f"\nTesting template: {template!r}")
        try:
            test_modelformmixin_should_not_raise_confusing_keyerror(template)
            print("  -> Test passed (no error)")
        except AssertionError as e:
            print(f"  -> AssertionError: Bug confirmed")
        except Exception as e:
            print(f"  -> Other error: {type(e).__name__}: {e}")

print("\n" + "=" * 50)
print("Running full Hypothesis test suite...")
print("=" * 50)

# Run the full test
try:
    test_modelformmixin_should_not_raise_confusing_keyerror()
    print("Full Hypothesis test passed - no issues found")
except Exception as e:
    print(f"Hypothesis test failed: {type(e).__name__}")
    print(str(e))