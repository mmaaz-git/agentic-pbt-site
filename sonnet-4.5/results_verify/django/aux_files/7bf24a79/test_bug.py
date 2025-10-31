#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.core.files.base import File

class FalsyButOpenFile:
    def __bool__(self):
        return False

    @property
    def closed(self):
        return False

falsy_file = FalsyButOpenFile()
file_obj = File(falsy_file)

print(f"Underlying file closed: {falsy_file.closed}")
print(f"FileProxyMixin closed: {file_obj.closed}")

# Test the assertion
try:
    assert file_obj.closed == False, f"Expected False, got {file_obj.closed}"
    print("Assertion passed: file_obj.closed == False")
except AssertionError as e:
    print(f"Assertion failed: {e}")