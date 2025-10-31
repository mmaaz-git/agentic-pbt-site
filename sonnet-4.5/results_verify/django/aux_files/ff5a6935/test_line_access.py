#!/usr/bin/env python3
"""Test if the bug causes issues with line access"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.template.backends.jinja2 import get_exception_info

class MockJinja2Exception:
    def __init__(self, lineno, source, message, filename):
        self.lineno = lineno
        self.source = source
        self.message = message
        self.filename = filename

# Test if empty source with lineno=1 causes errors
print("Test: Empty source with lineno=1")
try:
    exc = MockJinja2Exception(lineno=1, source='', message='test', filename='test.html')
    info = get_exception_info(exc)
    print(f"Success: {info}")
    print(f"  - during: '{info['during']}'")
    print(f"  - total: {info['total']}")
    print(f"  - source_lines: {info['source_lines']}")
except Exception as e:
    print(f"Error: {e}")

print("\nTest: Empty source with lineno=2")
try:
    exc = MockJinja2Exception(lineno=2, source='', message='test', filename='test.html')
    info = get_exception_info(exc)
    print(f"Success: {info}")
except Exception as e:
    print(f"Error: {e}")

print("\nTest: Whitespace-only source with lineno=1")
try:
    exc = MockJinja2Exception(lineno=1, source='   \n  ', message='test', filename='test.html')
    info = get_exception_info(exc)
    print(f"Success: {info}")
    print(f"  - during: '{info['during']}'")
    print(f"  - total: {info['total']}")
    print(f"  - source_lines: {info['source_lines']}")
except Exception as e:
    print(f"Error: {e}")