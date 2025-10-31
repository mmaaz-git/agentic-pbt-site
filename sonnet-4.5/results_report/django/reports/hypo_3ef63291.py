import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.conf import settings
settings.configure(USE_I18N=False)

from hypothesis import given, strategies as st, assume
from decimal import Decimal
from django.core.validators import DecimalValidator
from django.core.exceptions import ValidationError

@given(
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=0, max_value=20),
)
def test_decimal_validator_accepts_zero(max_digits, decimal_places):
    assume(decimal_places <= max_digits)

    validator = DecimalValidator(max_digits, decimal_places)

    decimal_value = Decimal("0.0")
    validator(decimal_value)

if __name__ == "__main__":
    test_decimal_validator_accepts_zero()