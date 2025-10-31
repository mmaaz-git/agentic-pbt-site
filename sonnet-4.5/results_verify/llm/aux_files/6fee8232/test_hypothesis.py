import django
from django.conf import settings

if not settings.configured:
    settings.configure(DEBUG=True, SECRET_KEY='test', USE_I18N=True)
    django.setup()

from hypothesis import given, strategies as st
import django.forms as forms


@given(st.booleans())
def test_integerfield_should_accept_booleans(b):
    """
    Property: IntegerField should accept boolean values since int() can be called on booleans.

    The IntegerField docstring states: "Validate that int() can be called on the input."
    Since int(True) == 1 and int(False) == 0 are valid Python operations,
    IntegerField should accept boolean values.
    """
    field = forms.IntegerField()

    try:
        result = field.clean(b)
        expected = int(b)
        assert result == expected, f"Expected {expected}, got {result}"
    except forms.ValidationError:
        assert False, f"IntegerField should accept boolean {b} since int({b}) == {int(b)}"

if __name__ == "__main__":
    test_integerfield_should_accept_booleans()
    print("Test completed")