#!/usr/bin/env python3
from django.core.mail import EmailMessage
from django.conf import settings

if not settings.configured:
    settings.configure(
        DEFAULT_CHARSET='utf-8',
        EMAIL_BACKEND='django.core.mail.backends.locmem.EmailBackend',
        DEFAULT_FROM_EMAIL='test@example.com',
    )

# Test 1: Basic reproduction - passing None as filename
print("Test 1: Reproducing the bug with None filename...")
try:
    msg = EmailMessage(
        subject="Test",
        body="Test",
        from_email="test@example.com",
        to=["to@example.com"],
    )
    msg.attach(filename=None, content=b"test content", mimetype=None)
    print("SUCCESS: No error occurred")
    message = msg.message()
    print(f"Message created successfully: {message is not None}")
except TypeError as e:
    print(f"FAILED with TypeError: {e}")
except Exception as e:
    print(f"FAILED with unexpected error: {e}")

print("\n" + "="*50 + "\n")

# Test 2: Testing with omitted filename (positional arguments)
print("Test 2: Testing with positional arguments...")
try:
    msg2 = EmailMessage(
        subject="Test2",
        body="Test2",
        from_email="test@example.com",
        to=["to@example.com"],
    )
    # Using positional arguments: filename, content, mimetype
    msg2.attach(None, b"test content", None)
    print("SUCCESS: No error occurred")
    message2 = msg2.message()
    print(f"Message created successfully: {message2 is not None}")
except TypeError as e:
    print(f"FAILED with TypeError: {e}")
except Exception as e:
    print(f"FAILED with unexpected error: {e}")

print("\n" + "="*50 + "\n")

# Test 3: What about empty string as filename?
print("Test 3: Testing with empty string as filename...")
try:
    msg3 = EmailMessage(
        subject="Test3",
        body="Test3",
        from_email="test@example.com",
        to=["to@example.com"],
    )
    msg3.attach(filename="", content=b"test content", mimetype=None)
    print("SUCCESS: No error occurred")
    message3 = msg3.message()
    print(f"Message created successfully: {message3 is not None}")
except Exception as e:
    print(f"FAILED with error: {e}")

print("\n" + "="*50 + "\n")

# Test 4: Normal usage with a valid filename
print("Test 4: Testing normal usage with valid filename...")
try:
    msg4 = EmailMessage(
        subject="Test4",
        body="Test4",
        from_email="test@example.com",
        to=["to@example.com"],
    )
    msg4.attach(filename="test.txt", content=b"test content", mimetype=None)
    print("SUCCESS: No error occurred")
    message4 = msg4.message()
    print(f"Message created successfully: {message4 is not None}")
except Exception as e:
    print(f"FAILED with error: {e}")