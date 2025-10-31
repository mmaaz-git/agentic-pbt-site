#!/usr/bin/env python3
"""Test script to reproduce the bug"""

from django.apps.config import AppConfig

print("Test 1: Trying to create AppConfig with entry='myapp.'")
try:
    entry = "myapp."
    config = AppConfig.create(entry)
    print(f"Success: Created config {config}")
except IndexError as e:
    print(f"IndexError occurred: {e}")
    import traceback
    traceback.print_exc()
except ImportError as e:
    print(f"ImportError occurred (expected): {e}")
except Exception as e:
    print(f"Other exception occurred: {type(e).__name__}: {e}")

print("\n" + "="*50 + "\n")

print("Test 2: Testing what rpartition does with 'myapp.'")
entry = "myapp."
mod_path, _, cls_name = entry.rpartition(".")
print(f"entry.rpartition('.') results:")
print(f"  mod_path = {repr(mod_path)}")
print(f"  cls_name = {repr(cls_name)}")
print(f"  bool(mod_path) = {bool(mod_path)}")
print(f"  bool(cls_name) = {bool(cls_name)}")
if cls_name:
    print(f"  cls_name[0] = {repr(cls_name[0])}")
else:
    print(f"  cls_name is empty, so cls_name[0] would raise IndexError")

print("\n" + "="*50 + "\n")

print("Test 3: Testing with other edge cases")
test_cases = [
    ".",
    "..",
    "...",
    "a.",
    ".a",
    "a.b.",
    "a..b",
    "",
]

for test_entry in test_cases:
    print(f"\nTesting entry: {repr(test_entry)}")
    mod_path, _, cls_name = test_entry.rpartition(".")
    print(f"  mod_path = {repr(mod_path)}, cls_name = {repr(cls_name)}")

    try:
        config = AppConfig.create(test_entry)
        print(f"  Result: Success - created {config}")
    except IndexError as e:
        print(f"  Result: IndexError - {e}")
    except ImportError as e:
        print(f"  Result: ImportError - {e}")
    except Exception as e:
        print(f"  Result: {type(e).__name__} - {e}")