import django
from django.conf import settings

if not settings.configured:
    settings.configure(DEBUG=True, SECRET_KEY='test')
    django.setup()

from django.forms.utils import ErrorDict, ErrorList
from django.forms.renderers import get_default_renderer

# Create an ErrorDict with a custom renderer
renderer = get_default_renderer()
error_dict = ErrorDict(
    {'field1': ErrorList(['Error 1', 'Error 2']), 'field2': ErrorList(['Error 3'])},
    renderer=renderer
)

# Verify the original has the renderer
print("Original ErrorDict has renderer attribute:", hasattr(error_dict, 'renderer'))
print("Original ErrorDict renderer value:", error_dict.renderer)

# Create a copy using the copy() method
copied = error_dict.copy()

# Check if the copy has the renderer attribute
print("\nCopied ErrorDict has renderer attribute:", hasattr(copied, 'renderer'))

# This will raise an AttributeError
try:
    print("Copied ErrorDict renderer value:", copied.renderer)
except AttributeError as e:
    print("AttributeError accessing copied.renderer:", e)

# Verify the data was copied correctly
print("\nOriginal ErrorDict data:", dict(error_dict))
print("Copied ErrorDict data:", dict(copied))

# Show that ErrorList correctly preserves renderer on copy
error_list = ErrorList(['Error 1', 'Error 2'], renderer=renderer)
copied_list = error_list.copy()
print("\nErrorList copy preserves renderer:", hasattr(copied_list, 'renderer'))
print("ErrorList copied renderer value:", copied_list.renderer)