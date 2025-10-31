#!/usr/bin/env python3
"""Test to reproduce the filebased backend empty messages bug"""

import sys
import os
import tempfile

# Configure Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')

import django
django.setup()

from django.core.mail.backends.filebased import EmailBackend as FileBasedBackend
from django.core.mail.backends.console import EmailBackend as ConsoleBackend
from django.core.mail.backends.dummy import EmailBackend as DummyBackend
from django.core.mail.backends.locmem import EmailBackend as LocmemBackend
from django.core.mail.backends.smtp import EmailBackend as SMTPBackend

def test_empty_messages_return_value():
    """Test that all backends return consistent values for empty messages"""

    print("Testing empty message list return values for all backends...")
    print("=" * 60)

    results = {}

    # Test filebased backend
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = FileBasedBackend(file_path=tmpdir)
        result = backend.send_messages([])
        results['filebased'] = result
        print(f"FileBasedBackend.send_messages([]): {result} (type: {type(result).__name__})")

    # Test console backend
    import io
    stream = io.StringIO()
    backend = ConsoleBackend(stream=stream)
    result = backend.send_messages([])
    results['console'] = result
    print(f"ConsoleBackend.send_messages([]):   {result} (type: {type(result).__name__})")

    # Test dummy backend
    backend = DummyBackend()
    result = backend.send_messages([])
    results['dummy'] = result
    print(f"DummyBackend.send_messages([]):     {result} (type: {type(result).__name__})")

    # Test locmem backend
    backend = LocmemBackend()
    result = backend.send_messages([])
    results['locmem'] = result
    print(f"LocmemBackend.send_messages([]):    {result} (type: {type(result).__name__})")

    # Test SMTP backend (without connecting)
    backend = SMTPBackend(host='localhost', fail_silently=True)
    result = backend.send_messages([])
    results['smtp'] = result
    print(f"SMTPBackend.send_messages([]):      {result} (type: {type(result).__name__})")

    print("=" * 60)
    print("\nAnalysis:")

    # Check consistency
    inconsistent = []
    for name, value in results.items():
        if value != 0:
            inconsistent.append(f"  - {name}: returns {value} instead of 0")

    if inconsistent:
        print("INCONSISTENT BACKENDS (should return 0):")
        for issue in inconsistent:
            print(issue)
        print("\nThis violates the documented API contract.")
    else:
        print("All backends consistently return 0 for empty messages.")

    return results

def test_hypothesis_property():
    """Run the property-based test from the bug report"""
    from hypothesis import given, strategies as st

    print("\n" + "=" * 60)
    print("Running property-based test from bug report...")
    print("=" * 60)

    @given(st.booleans())
    def test_filebased_backend_empty_messages_returns_int(fail_silently):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileBasedBackend(fail_silently=fail_silently, file_path=tmpdir)
            result = backend.send_messages([])
            assert isinstance(result, int), f"Expected int, got {type(result).__name__}: {result}"
            assert result == 0, f"Expected 0 for empty messages, got {result}"

    try:
        test_filebased_backend_empty_messages_returns_int()
        print("Property test PASSED")
    except AssertionError as e:
        print(f"Property test FAILED: {e}")
        return False

    return True

def check_inheritance():
    """Check the inheritance structure"""
    print("\n" + "=" * 60)
    print("Inheritance Analysis:")
    print("=" * 60)

    from django.core.mail.backends.base import BaseEmailBackend

    print(f"FileBasedBackend inherits from: {FileBasedBackend.__bases__}")
    print(f"ConsoleBackend inherits from: {ConsoleBackend.__bases__}")

    # Check if filebased inherits send_messages from console
    print(f"\nFileBasedBackend.send_messages method defined in: {FileBasedBackend.send_messages.__qualname__}")
    print(f"ConsoleBackend.send_messages method defined in: {ConsoleBackend.send_messages.__qualname__}")

    # Look at the source
    import inspect
    console_source = inspect.getsource(ConsoleBackend.send_messages)
    print("\nConsoleBackend.send_messages source (line 31):")
    for i, line in enumerate(console_source.split('\n')[0:5], 1):
        if 'return' in line and 'email_messages' in line:
            print(f"  Line {i}: {line}  <-- BUG HERE")
        else:
            print(f"  Line {i}: {line}")

if __name__ == "__main__":
    results = test_empty_messages_return_value()
    property_passed = test_hypothesis_property()
    check_inheritance()

    print("\n" + "=" * 60)
    print("CONCLUSION:")
    print("=" * 60)
    if results['filebased'] is None and results['console'] is None:
        print("BUG CONFIRMED: Both console and filebased backends return None")
        print("instead of 0 for empty message lists, violating the documented")
        print("API contract that states they should 'return the number of")
        print("email messages sent'.")
    else:
        print("Bug not reproduced as described.")