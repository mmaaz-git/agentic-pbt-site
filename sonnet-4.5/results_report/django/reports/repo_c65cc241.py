#!/usr/bin/env python3
"""Minimal reproduction of Django filebased email backend None path crash"""

from django.conf import settings
from django.core.mail.backends.filebased import EmailBackend

# Configure Django with EMAIL_FILE_PATH set to None
if not settings.configured:
    settings.configure(EMAIL_FILE_PATH=None, DEFAULT_CHARSET='utf-8')

# This should crash with TypeError when both file_path and EMAIL_FILE_PATH are None
try:
    backend = EmailBackend(file_path=None)
    print("ERROR: Expected TypeError but backend was created successfully")
except TypeError as e:
    print(f"Caught expected TypeError: {e}")
    import traceback
    traceback.print_exc()