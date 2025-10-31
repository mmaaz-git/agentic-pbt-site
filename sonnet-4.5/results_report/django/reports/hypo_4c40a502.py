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
from hypothesis import given, strategies as st, example

@given(
    parts=st.lists(st.binary(min_size=0, max_size=20), min_size=2, max_size=5),
    path_info=st.binary(min_size=0, max_size=50)
)
@example(parts=[b'', b'\x80'], path_info=b'')  # The specific failing case
def test_get_script_name_with_multiple_slashes(parts, path_info):
    script_url = b'//'.join(parts)

    environ = {
        'SCRIPT_URL': script_url.decode('latin1', errors='replace'),
        'PATH_INFO': path_info.decode('latin1', errors='replace'),
        'SCRIPT_NAME': ''
    }

    try:
        script_name = get_script_name(environ)
        assert '//' not in script_name
        print(f"✓ Test passed for parts={parts}, path_info={path_info}")
    except UnicodeDecodeError as e:
        print(f"✗ UnicodeDecodeError with parts={parts}, path_info={path_info}")
        print(f"  Error: {e}")
        raise

# Run the test
if __name__ == "__main__":
    print("Running Hypothesis test with specific failing example...")
    test_get_script_name_with_multiple_slashes()