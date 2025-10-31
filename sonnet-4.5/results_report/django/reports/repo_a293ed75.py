import django
from django.conf import settings

if not settings.configured:
    settings.configure(INSTALLED_APPS=[], SECRET_KEY='test', USE_TZ=True)
    django.setup()

from django.db import models
from django.db.migrations.operations import AddField

# Create a CharField
field = models.CharField(max_length=100)

# Create an AddField operation with field name "firstName"
op = AddField(model_name="User", name="firstName", field=field)

# Create another AddField operation with field name in uppercase "FIRSTNAME"
op2 = AddField(model_name="User", name="FIRSTNAME", field=field)

# Test is_same_field_operation - expects True due to case-insensitive comparison
print(f"op.is_same_field_operation(op2): {op.is_same_field_operation(op2)}")

# Test references_field with uppercase name - expects True but returns False
print(f"op.references_field('User', 'FIRSTNAME', 'app'): {op.references_field('User', 'FIRSTNAME', 'app')}")

# The inconsistency: is_same_field_operation says they're the same field,
# but references_field says the first operation doesn't reference the second
print("\nInconsistency detected:")
print(f"  is_same_field_operation returns: {op.is_same_field_operation(op2)}")
print(f"  references_field returns: {op.references_field('User', 'FIRSTNAME', 'app')}")
print(f"  Expected: Both should return True")