#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.core.handlers.wsgi import LimitedStream
from io import BytesIO

# Test LimitedStream with negative limit
print("Testing LimitedStream with negative limit:")

data = b"Hello, World! This is test data."
stream = BytesIO(data)

# Create LimitedStream with negative limit
limited = LimitedStream(stream, -1)

print(f"Limit: {limited.limit}")
print(f"Initial position: {limited._pos}")

# Try to read
result1 = limited.read()
print(f"Read all: {result1}")
print(f"Position after read: {limited._pos}")

# Reset and try readline
stream2 = BytesIO(data)
limited2 = LimitedStream(stream2, -1)
result2 = limited2.readline()
print(f"Readline: {result2}")

# Test with positive limit for comparison
print("\nTesting LimitedStream with positive limit (10):")
stream3 = BytesIO(data)
limited3 = LimitedStream(stream3, 10)
result3 = limited3.read()
print(f"Read with limit 10: {result3}")
print(f"Length read: {len(result3)}")