#!/usr/bin/env python3
"""Minimal reproduction of Django email backend empty messages bug."""

import tempfile
import django
from django.conf import settings

# Configure Django settings
settings.configure(
    EMAIL_HOST='localhost',
    EMAIL_PORT=25,
    EMAIL_HOST_USER='',
    EMAIL_HOST_PASSWORD='',
    EMAIL_USE_TLS=False,
    EMAIL_USE_SSL=False,
    DEFAULT_CHARSET='utf-8',
)
django.setup()

from django.core.mail.backends.filebased import EmailBackend as FileBasedBackend
from django.core.mail.backends.smtp import EmailBackend as SMTPBackend
from django.core.mail.backends.locmem import EmailBackend as LocmemBackend
from django.core.mail.backends.console import EmailBackend as ConsoleBackend

# Test with empty message list
empty_messages = []

# Create temporary directory for file-based backend
with tempfile.TemporaryDirectory() as tmpdir:
    # Initialize backends
    filebased_backend = FileBasedBackend(file_path=tmpdir)
    smtp_backend = SMTPBackend()
    locmem_backend = LocmemBackend()
    console_backend = ConsoleBackend()

    # Call send_messages with empty list
    filebased_result = filebased_backend.send_messages(empty_messages)
    smtp_result = smtp_backend.send_messages(empty_messages)
    locmem_result = locmem_backend.send_messages(empty_messages)
    console_result = console_backend.send_messages(empty_messages)

    # Display results
    print(f"Filebased backend result: {filebased_result!r} (type: {type(filebased_result).__name__})")
    print(f"Console backend result: {console_result!r} (type: {type(console_result).__name__})")
    print(f"SMTP backend result: {smtp_result!r} (type: {type(smtp_result).__name__})")
    print(f"Locmem backend result: {locmem_result!r} (type: {type(locmem_result).__name__})")

    print("\n--- Type inconsistency ---")
    print(f"Filebased returns: {type(filebased_result)}")
    print(f"Console returns: {type(console_result)}")
    print(f"SMTP returns: {type(smtp_result)}")
    print(f"Locmem returns: {type(locmem_result)}")

    print("\n--- Attempting numeric comparison ---")
    try:
        result = filebased_result > 0
        print(f"filebased_result > 0 = {result}")
    except TypeError as e:
        print(f"TypeError when comparing filebased_result > 0: {e}")

    try:
        result = console_result > 0
        print(f"console_result > 0 = {result}")
    except TypeError as e:
        print(f"TypeError when comparing console_result > 0: {e}")

    print("\n--- Backend substitutability broken ---")
    print("Cannot transparently swap filebased/console with smtp/locmem backends")
    print("due to different return types for empty message lists.")