import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')

from django.views.i18n import JavaScriptCatalog
from unittest.mock import Mock

# Create a JavaScriptCatalog instance
catalog = JavaScriptCatalog()

# Mock the translation object with a malformed Plural-Forms header
# This header has nplurals but is missing the plural= part
catalog.translation = Mock()
catalog.translation._catalog = {
    "": "Plural-Forms: nplurals=2;"
}

# Try to get the plural - this should crash with IndexError
try:
    result = catalog.get_plural()
    print(f"Result: {result}")
except IndexError as e:
    print(f"IndexError: {e}")
    import traceback
    traceback.print_exc()