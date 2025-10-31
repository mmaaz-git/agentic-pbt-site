import django
from django.conf import settings
settings.configure(USE_I18N=True, USE_TZ=False)
django.setup()

from django.db.models.fields import Field

# Check the Field class docstring
print("Field class docstring:")
print("=" * 60)
print(Field.__doc__)
print("\n")

# Check the validate method docstring
print("Field.validate method docstring:")
print("=" * 60)
print(Field.validate.__doc__)
print("\n")

# Check what empty_values contains
print("Field.empty_values:")
print("=" * 60)
field = Field()
print(field.empty_values)