#!/usr/bin/env python3
"""Reproduce the reported bug in django.template.backends.jinja2.get_exception_info"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.template.backends.jinja2 import get_exception_info

class MockJinja2Exception:
    def __init__(self, lineno, source, message, filename):
        self.lineno = lineno
        self.source = source
        self.message = message
        self.filename = filename

# Test with empty source
print("=" * 60)
print("Test 1: Empty source string")
exc = MockJinja2Exception(lineno=1, source='', message='test', filename='test.html')
info = get_exception_info(exc)
print(f"Empty source has {info['total']} lines")
print(f"Expected: 0 lines")
print(f"Result: {'BUG' if info['total'] != 0 else 'OK'}")

# Test what Python's split does with empty string
print("\n" + "=" * 60)
print("Test 2: Python behavior with empty strings")
empty_str = ''
split_result = empty_str.split('\n')
print(f"''.split('\\n') = {split_result}")
print(f"Length: {len(split_result)}")

stripped = empty_str.strip()
split_stripped = stripped.split('\n')
print(f"''.strip().split('\\n') = {split_stripped}")
print(f"Length: {len(split_stripped)}")

# Test with whitespace-only source
print("\n" + "=" * 60)
print("Test 3: Whitespace-only source")
exc2 = MockJinja2Exception(lineno=1, source='   \n  \t  ', message='test', filename='test.html')
info2 = get_exception_info(exc2)
print(f"Whitespace-only source has {info2['total']} lines")
print(f"After strip: '{exc2.source.strip()}'")
print(f"Expected: 0 lines (since it strips to empty)")
print(f"Result: {'BUG' if info2['total'] != 0 else 'OK'}")

# Test with single line
print("\n" + "=" * 60)
print("Test 4: Single line source")
exc3 = MockJinja2Exception(lineno=1, source='hello', message='test', filename='test.html')
info3 = get_exception_info(exc3)
print(f"Single line source has {info3['total']} lines")
print(f"Expected: 1 line")
print(f"Result: {'OK' if info3['total'] == 1 else 'BUG'}")

# Test with multiple lines
print("\n" + "=" * 60)
print("Test 5: Multiple line source")
exc4 = MockJinja2Exception(lineno=2, source='line1\nline2\nline3', message='test', filename='test.html')
info4 = get_exception_info(exc4)
print(f"Three-line source has {info4['total']} lines")
print(f"Expected: 3 lines")
print(f"Result: {'OK' if info4['total'] == 3 else 'BUG'}")