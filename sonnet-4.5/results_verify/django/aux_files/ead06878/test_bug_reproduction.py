#!/usr/bin/env python3
"""Test reproduction for the get_script_name bug report"""

import sys
import traceback

# First test: The Hypothesis test case
def test_hypothesis_case():
    """Test the hypothesis case with the failing input"""
    print("=" * 60)
    print("Testing Hypothesis Case")
    print("=" * 60)

    try:
        # Setup Django
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

        # Create the failing input from the hypothesis test
        parts = [b'', b'\x80']
        path_info = b''

        script_url = b'//'.join(parts)
        print(f"Script URL bytes: {script_url!r}")

        # Decode with latin1 as specified
        script_url_str = script_url.decode('latin1', errors='replace')
        path_info_str = path_info.decode('latin1', errors='replace')

        print(f"Script URL string (latin1 decoded): {script_url_str!r}")

        environ = {
            'SCRIPT_URL': script_url_str,
            'PATH_INFO': path_info_str,
            'SCRIPT_NAME': ''
        }

        print(f"Environ: {environ}")

        # This should crash according to the bug report
        result = get_script_name(environ)
        print(f"Result: {result!r}")
        print("SUCCESS: No crash occurred!")
        return True

    except UnicodeDecodeError as e:
        print(f"FAILURE: UnicodeDecodeError occurred!")
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        return False

def test_direct_reproduction():
    """Test the direct reproduction code from the bug report"""
    print("\n" + "=" * 60)
    print("Testing Direct Reproduction")
    print("=" * 60)

    try:
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
            'SCRIPT_URL': '\x80',  # This is a latin-1 character
            'PATH_INFO': '',
            'SCRIPT_NAME': ''
        }

        print(f"Environ: {environ}")

        result = get_script_name(environ)
        print(f"Result: {result!r}")
        print("SUCCESS: No crash occurred!")
        return True

    except UnicodeDecodeError as e:
        print(f"FAILURE: UnicodeDecodeError occurred!")
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        return False

def test_similar_functions():
    """Test how similar functions handle the same input"""
    print("\n" + "=" * 60)
    print("Testing Similar Functions with Same Input")
    print("=" * 60)

    try:
        import django
        from django.conf import settings
        if not settings.configured:
            settings.configure(
                DEBUG=True,
                SECRET_KEY='test-secret-key',
                FORCE_SCRIPT_NAME=None,
            )
            django.setup()

        from django.core.handlers.wsgi import get_path_info, get_str_from_wsgi, get_script_name

        # Test with invalid UTF-8 byte sequence
        environ = {
            'SCRIPT_URL': '\x80',  # Invalid UTF-8 when re-encoded
            'PATH_INFO': '\x80',
            'SCRIPT_NAME': '\x80',
            'TEST_VALUE': '\x80'
        }

        print("Testing get_path_info with invalid UTF-8:")
        try:
            result = get_path_info(environ)
            print(f"  SUCCESS: get_path_info returned: {result!r}")
        except Exception as e:
            print(f"  FAILURE: get_path_info crashed: {e}")

        print("\nTesting get_str_from_wsgi with invalid UTF-8:")
        try:
            result = get_str_from_wsgi(environ, 'TEST_VALUE', '')
            print(f"  SUCCESS: get_str_from_wsgi returned: {result!r}")
        except Exception as e:
            print(f"  FAILURE: get_str_from_wsgi crashed: {e}")

        print("\nTesting get_script_name with invalid UTF-8:")
        try:
            result = get_script_name(environ)
            print(f"  SUCCESS: get_script_name returned: {result!r}")
        except Exception as e:
            print(f"  FAILURE: get_script_name crashed: {e}")
            traceback.print_exc()

    except Exception as e:
        print(f"Setup error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_hypothesis_case()
    test_direct_reproduction()
    test_similar_functions()