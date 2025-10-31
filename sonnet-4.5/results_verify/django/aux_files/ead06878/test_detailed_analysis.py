#!/usr/bin/env python3
"""Detailed analysis of the bug"""

import django
from django.conf import settings
if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY='test-secret-key',
        FORCE_SCRIPT_NAME=None,
    )
    django.setup()

from django.core.handlers.wsgi import get_bytes_from_wsgi

# Test what happens when we get bytes from WSGI
environ = {
    'SCRIPT_URL': '\x80',  # This is ISO-8859-1 encoded in the WSGI environ
    'PATH_INFO': '\x80',
    'SCRIPT_NAME': ''
}

print("Testing get_bytes_from_wsgi behavior:")
print("=" * 60)

# What does get_bytes_from_wsgi return?
script_url_bytes = get_bytes_from_wsgi(environ, "SCRIPT_URL", "")
print(f"SCRIPT_URL as bytes: {script_url_bytes!r}")

path_info_bytes = get_bytes_from_wsgi(environ, "PATH_INFO", "")
print(f"PATH_INFO as bytes: {path_info_bytes!r}")

# Test decoding with different strategies
print("\n" + "=" * 60)
print("Attempting different decode strategies:")

print("\n1. bare decode() - what get_script_name uses:")
try:
    result = script_url_bytes.decode()
    print(f"   Success: {result!r}")
except UnicodeDecodeError as e:
    print(f"   FAILS: {e}")

print("\n2. decode(errors='replace') - what get_str_from_wsgi uses:")
try:
    result = script_url_bytes.decode(errors='replace')
    print(f"   Success: {result!r}")
except UnicodeDecodeError as e:
    print(f"   Fails: {e}")

print("\n3. repercent_broken_unicode - what get_path_info uses:")
try:
    from django.utils.encoding import repercent_broken_unicode
    result = repercent_broken_unicode(path_info_bytes).decode()
    print(f"   Success: {result!r}")
except Exception as e:
    print(f"   Fails: {e}")

# Test real-world scenario
print("\n" + "=" * 60)
print("Real-world scenario test:")
print("A client sends a malformed URL with invalid UTF-8 bytes")
print("The WSGI server decodes it as ISO-8859-1 (as required by PEP 3333)")
print("Django tries to decode it as UTF-8 and crashes")
print()

# Simulate what a WSGI server would do
malicious_url = b'/script\x80name'  # Invalid UTF-8
print(f"Original malicious bytes: {malicious_url!r}")

# WSGI server decodes with ISO-8859-1 (always succeeds)
wsgi_string = malicious_url.decode('iso-8859-1')
print(f"WSGI environ value (ISO-8859-1 decoded): {wsgi_string!r}")

# Django re-encodes to get bytes back
django_bytes = wsgi_string.encode('iso-8859-1')
print(f"Django re-encoded bytes: {django_bytes!r}")

# Django tries to decode as UTF-8
print("\nDjango tries script_name.decode():")
try:
    result = django_bytes.decode()
    print(f"   Success: {result!r}")
except UnicodeDecodeError as e:
    print(f"   CRASH: {e}")