import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')

from hypothesis import given, strategies as st, assume
from django.views.i18n import JavaScriptCatalog
from unittest.mock import Mock

@given(st.text(min_size=1))
def test_get_plural_never_crashes(plural_forms_value):
    assume("plural=" not in plural_forms_value or not any(
        part.strip().startswith("plural=")
        for part in plural_forms_value.split(";")
    ))

    catalog = JavaScriptCatalog()
    catalog.translation = Mock()
    catalog.translation._catalog = {
        "": f"Plural-Forms: {plural_forms_value}"
    }

    # This should not crash
    try:
        result = catalog.get_plural()
        print(f"Passed with input '{plural_forms_value}' -> result: {result}")
    except IndexError as e:
        print(f"Failed with input '{plural_forms_value}' -> IndexError: {e}")
        raise
    except Exception as e:
        print(f"Failed with input '{plural_forms_value}' -> {type(e).__name__}: {e}")
        raise

print("Running property-based test...")
test_get_plural_never_crashes()