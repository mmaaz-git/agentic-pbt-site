import django
from django.conf import settings
settings.configure(USE_I18N=True, USE_TZ=False)
django.setup()

from hypothesis import given, strategies as st, example
from django.db.models.fields import CharField
from django.core.exceptions import ValidationError
import pytest


@given(
    st.lists(st.tuples(st.text(), st.text()), min_size=1),
    st.text()
)
@example(choices=[('', '')], value='')  # The specific failing case
def test_charfield_validates_choices(choices, value):
    """
    Property: CharField should accept values in choices, reject others.
    """
    field = CharField(choices=choices)
    choice_values = {choice[0] for choice in choices}

    if value in choice_values:
        try:
            field.validate(value, None)
            print(f"✓ Validation passed for value='{value}' with choices containing '{value}'")
        except ValidationError as e:
            print(f"✗ Validation FAILED for value='{value}' with choices containing '{value}'")
            print(f"  Error: {e}")
            raise AssertionError(f"Field should accept value '{value}' that's in choices")
    else:
        try:
            field.validate(value, None)
            print(f"✗ Validation PASSED but should have failed for value='{value}' not in choices")
            raise AssertionError(f"Field should reject value '{value}' not in choices")
        except ValidationError:
            print(f"✓ Correctly rejected value='{value}' not in choices")

# Run the test
print("Running hypothesis test...")
try:
    test_charfield_validates_choices()
    print("\nAll tests passed!")
except Exception as e:
    print(f"\nTest failed with error: {e}")