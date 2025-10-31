import django
from django.conf import settings
if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY='test-secret-key',
        FORCE_SCRIPT_NAME=None,
    )
    django.setup()

from django.core.handlers.wsgi import get_script_name, get_path_info, get_str_from_wsgi

# Test the same invalid UTF-8 byte (0x80) with all three functions
# This simulates what happens when WSGI decodes bytes with latin-1
test_input = '\x80'  # This is a latin-1 decoded byte that's invalid UTF-8

print("Testing how different functions handle invalid UTF-8 (byte 0x80):\n")

# Test get_path_info with PATH_INFO containing invalid UTF-8
print("1. get_path_info() with PATH_INFO='\\x80':")
environ = {
    'PATH_INFO': test_input,
}
try:
    result = get_path_info(environ)
    print(f"   SUCCESS: Returns {repr(result)}")
except Exception as e:
    print(f"   FAILURE: {type(e).__name__}: {e}")

# Test get_str_from_wsgi with a key containing invalid UTF-8
print("\n2. get_str_from_wsgi() with TEST_KEY='\\x80':")
environ = {
    'TEST_KEY': test_input
}
try:
    result = get_str_from_wsgi(environ, 'TEST_KEY', '')
    print(f"   SUCCESS: Returns {repr(result)}")
except Exception as e:
    print(f"   FAILURE: {type(e).__name__}: {e}")

# Test get_script_name with SCRIPT_URL containing invalid UTF-8
print("\n3. get_script_name() with SCRIPT_URL='\\x80':")
environ = {
    'SCRIPT_URL': test_input,
    'PATH_INFO': '',  # empty to ensure we take the SCRIPT_URL path
    'SCRIPT_NAME': ''
}
try:
    result = get_script_name(environ)
    print(f"   SUCCESS: Returns {repr(result)}")
except Exception as e:
    print(f"   FAILURE: {type(e).__name__}: {e}")

# Also test with a longer path to ensure it's the script_url path causing the issue
print("\n4. get_script_name() with SCRIPT_URL='/path\\x80' (longer path):")
environ = {
    'SCRIPT_URL': '/path' + test_input,
    'PATH_INFO': '',
    'SCRIPT_NAME': ''
}
try:
    result = get_script_name(environ)
    print(f"   SUCCESS: Returns {repr(result)}")
except Exception as e:
    print(f"   FAILURE: {type(e).__name__}: {e}")

print("\nConclusion: get_script_name() crashes on invalid UTF-8 while other functions handle it gracefully.")