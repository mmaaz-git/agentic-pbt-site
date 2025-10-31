import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.conf import settings
settings.configure(USE_I18N=False)

from decimal import Decimal
from django.core.validators import DecimalValidator
from django.core.exceptions import ValidationError

validator = DecimalValidator(max_digits=1, decimal_places=0)

print("Testing DecimalValidator(max_digits=1, decimal_places=0)")
print("="*60)

# Test Decimal('0')
print("\nDecimal('0'):", end=" ")
try:
    validator(Decimal("0"))
    print("PASSED ✓")
except ValidationError as e:
    print(f"FAILED ✗ - Error: {e}")

# Test Decimal('0.0')
print("Decimal('0.0'):", end=" ")
try:
    validator(Decimal("0.0"))
    print("PASSED ✓")
except ValidationError as e:
    print(f"FAILED ✗ - Error: {e}")

# Test Decimal('0.00')
print("Decimal('0.00'):", end=" ")
try:
    validator(Decimal("0.00"))
    print("PASSED ✓")
except ValidationError as e:
    print(f"FAILED ✗ - Error: {e}")

print("\n" + "="*60)
print("Mathematical equivalence check:")
print(f"Decimal('0') == Decimal('0.0'): {Decimal('0') == Decimal('0.0')}")
print(f"Decimal('0') == Decimal('0.00'): {Decimal('0') == Decimal('0.00')}")

print("\n" + "="*60)
print("Internal representation analysis:")
print(f"Decimal('0') internal: {Decimal('0').as_tuple()}")
print(f"Decimal('0.0') internal: {Decimal('0.0').as_tuple()}")
print(f"Decimal('0.00') internal: {Decimal('0.00').as_tuple()}")