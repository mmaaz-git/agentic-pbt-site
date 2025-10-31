#!/usr/bin/env python3
"""Test to reproduce the generator exhaustion bug"""

import os
import django
from django.conf import settings

# Configure Django settings
if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY='fake-key-for-testing',
        DEFAULT_CHARSET='utf-8',
        EMAIL_BACKEND='django.core.mail.backends.dummy.EmailBackend',
    )
    django.setup()

# First, let's test the actual bug claim
from django.core.mail.backends.dummy import EmailBackend

print("Testing generator exhaustion bug...")
print("-" * 50)

backend = EmailBackend()

exhausted = False
def message_generator():
    global exhausted
    for i in range(3):
        print(f"Yielding message {i}")
        yield object()
    exhausted = True
    print("Generator exhausted!")

gen = message_generator()
print("Created generator, exhausted =", exhausted)

result = backend.send_messages(gen)
print(f"Result: {result}")
print(f"Exhausted: {exhausted}")

# Verify the assertion
if exhausted:
    print("✓ Bug confirmed: Generator was exhausted by dummy backend")
else:
    print("✗ Bug NOT confirmed: Generator was not exhausted")

print()
print("Testing with a list instead of generator...")
print("-" * 50)

# Test with a list to see expected behavior
messages = [object() for _ in range(3)]
result2 = backend.send_messages(messages)
print(f"Result with list: {result2}")

print()
print("Testing other backends with generators...")
print("-" * 50)

# Test console backend with generator
from django.core.mail.backends.console import EmailBackend as ConsoleBackend
import io
from django.core.mail.message import EmailMessage

console_backend = ConsoleBackend(stream=io.StringIO())

exhausted2 = False
def email_generator():
    global exhausted2
    for i in range(3):
        msg = EmailMessage(
            subject=f"Test {i}",
            body=f"Body {i}",
            from_email="test@example.com",
            to=["recipient@example.com"],
        )
        print(f"Yielding email message {i}")
        yield msg
    exhausted2 = True
    print("Email generator exhausted!")

gen2 = email_generator()
print("Created email generator, exhausted2 =", exhausted2)

result3 = console_backend.send_messages(gen2)
print(f"Console backend result: {result3}")
print(f"Email generator exhausted: {exhausted2}")

if exhausted2:
    print("✓ Console backend also exhausts generators (expected)")
else:
    print("✗ Console backend does not exhaust generators")

print()
print("Testing the proposed fix...")
print("-" * 50)

# Test if the proposed fix would work
class FixedDummyBackend(EmailBackend):
    def send_messages(self, email_messages):
        return sum(1 for _ in email_messages)

fixed_backend = FixedDummyBackend()

exhausted3 = False
def message_generator3():
    global exhausted3
    for i in range(3):
        print(f"Yielding message {i} for fixed backend")
        yield object()
    exhausted3 = True
    print("Generator 3 exhausted!")

gen3 = message_generator3()
print("Created generator3, exhausted3 =", exhausted3)

result4 = fixed_backend.send_messages(gen3)
print(f"Fixed backend result: {result4}")
print(f"Generator3 exhausted: {exhausted3}")

if exhausted3:
    print("✓ Fixed backend also exhausts generators")
else:
    print("✗ Fixed backend does not exhaust generators")