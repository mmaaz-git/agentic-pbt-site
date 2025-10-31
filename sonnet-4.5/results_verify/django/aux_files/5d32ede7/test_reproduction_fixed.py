"""Test to reproduce the Django mail filebased backend bug"""

import os
import sys

# Configure Django settings
import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY='test-secret-key',
        EMAIL_HOST='localhost',
        EMAIL_PORT=25,
        EMAIL_USE_TLS=False,
        INSTALLED_APPS=[],
    )
    django.setup()

from django.core.mail.backends.filebased import EmailBackend
from django.core.mail.backends.smtp import EmailBackend as SMTPBackend
from django.core.mail.backends.locmem import EmailBackend as LocmemBackend
from django.core.mail.backends.console import EmailBackend as ConsoleBackend
import tempfile

# Test 1: Simple reproduction
print("=== Test 1: Simple Reproduction ===")
with tempfile.TemporaryDirectory() as tmpdir:
    filebased_backend = EmailBackend(file_path=tmpdir)
    smtp_backend = SMTPBackend()
    console_backend = ConsoleBackend()

    filebased_result = filebased_backend.send_messages([])
    smtp_result = smtp_backend.send_messages([])
    console_result = console_backend.send_messages([])

    print(f"Filebased result: {filebased_result!r} (type: {type(filebased_result).__name__})")
    print(f"SMTP result: {smtp_result!r} (type: {type(smtp_result).__name__})")
    print(f"Console result: {console_result!r} (type: {type(console_result).__name__})")

    # Check for type mismatch
    if type(filebased_result) != type(smtp_result):
        print(f"TYPE MISMATCH: {type(filebased_result)} vs {type(smtp_result)}")

    # Try comparison operation
    try:
        result = filebased_result > 0
        print(f"Comparison filebased_result > 0: {result}")
    except TypeError as e:
        print(f"TypeError on comparison: {e}")

print("\n=== Test 2: Multiple Backend Consistency ===")
with tempfile.TemporaryDirectory() as tmpdir:
    filebased = EmailBackend(file_path=tmpdir)
    smtp = SMTPBackend()
    locmem = LocmemBackend()
    console = ConsoleBackend()

    filebased_result = filebased.send_messages([])
    smtp_result = smtp.send_messages([])
    locmem_result = locmem.send_messages([])
    console_result = console.send_messages([])

    print(f"Filebased: {filebased_result!r}")
    print(f"SMTP: {smtp_result!r}")
    print(f"Locmem: {locmem_result!r}")
    print(f"Console: {console_result!r}")

    # Check type consistency
    if filebased_result is None:
        print("ISSUE: Filebased returns None for empty messages")

    if console_result is None:
        print("ISSUE: Console returns None for empty messages")

    if smtp_result == 0:
        print("SMTP correctly returns 0 for empty messages")

    if locmem_result == 0:
        print("Locmem correctly returns 0 for empty messages")

    # Check if all are equal
    if filebased_result == smtp_result == locmem_result == console_result:
        print("All backends return same value")
    else:
        print("INCONSISTENCY: Backends return different values")

print("\n=== Test 3: Hypothesis Test from Bug Report ===")
from hypothesis import given, strategies as st

@given(st.just([]))
def test_empty_message_consistency_filebased(empty_messages):
    with tempfile.TemporaryDirectory() as tmpdir:
        filebased = EmailBackend(file_path=tmpdir)
        smtp = SMTPBackend()
        locmem = LocmemBackend()

        filebased_result = filebased.send_messages(empty_messages)
        smtp_result = smtp.send_messages(empty_messages)
        locmem_result = locmem.send_messages(empty_messages)

        print(f"Testing with empty_messages={empty_messages}")
        print(f"  Filebased: {filebased_result!r}")
        print(f"  SMTP: {smtp_result!r}")
        print(f"  Locmem: {locmem_result!r}")

        assert isinstance(filebased_result, int), \
            f"Expected int, got {type(filebased_result).__name__}"
        assert filebased_result == smtp_result == locmem_result, \
            f"Inconsistent return values: filebased={filebased_result}, smtp={smtp_result}, locmem={locmem_result}"

try:
    test_empty_message_consistency_filebased()
    print("Hypothesis test PASSED")
except AssertionError as e:
    print(f"Hypothesis test FAILED: {e}")