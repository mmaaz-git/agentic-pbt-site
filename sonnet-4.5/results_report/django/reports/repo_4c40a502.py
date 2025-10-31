import django
from django.conf import settings
if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY='test-secret-key',
        FORCE_SCRIPT_NAME=None,
    )
    django.setup()

from django.core.handlers.wsgi import get_script_name

environ = {
    'SCRIPT_URL': '\x80',
    'PATH_INFO': '',
    'SCRIPT_NAME': ''
}

try:
    result = get_script_name(environ)
    print(f"Success: get_script_name returned: {repr(result)}")
except UnicodeDecodeError as e:
    print(f"UnicodeDecodeError: {e}")
    print(f"  Error at position {e.start}: byte {hex(e.object[e.start])}")
    import traceback
    traceback.print_exc()