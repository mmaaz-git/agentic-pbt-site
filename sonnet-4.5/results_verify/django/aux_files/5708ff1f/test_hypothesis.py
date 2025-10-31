import django
from django.conf import settings
settings.configure(USE_I18N=True, USE_TZ=False)
django.setup()

from hypothesis import given, strategies as st
from django.db.models.fields import CharField
from django.core.exceptions import ValidationError
import pytest


@given(
    st.lists(st.tuples(st.text(), st.text()), min_size=1),
    st.text()
)
def test_charfield_validates_choices(choices, value):
    """
    Property: CharField should accept values in choices, reject others.
    """
    field = CharField(choices=choices)
    choice_values = {choice[0] for choice in choices}

    if value in choice_values:
        field.validate(value, None)
    else:
        with pytest.raises(ValidationError):
            field.validate(value, None)

# Test the specific failing case
print("Testing specific case: choices=[('', '')], value=''")
try:
    test_charfield_validates_choices(choices=[('', '')], value='')
    print("Test passed for empty string choice")
except Exception as e:
    print(f"Test failed: {e}")