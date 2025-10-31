#!/usr/bin/env python3
"""Test the DistutilsInfo.merge aliasing bug"""

from Cython.Build.Dependencies import DistutilsInfo

print("=" * 60)
print("Testing DistutilsInfo.merge list aliasing bug")
print("=" * 60)

# Basic reproduction test
print("\n1. Basic test from bug report:")
info1 = DistutilsInfo()
info2 = DistutilsInfo()
info2.values['libraries'] = ['lib1', 'lib2']

print(f"info1.values before merge: {info1.values}")
print(f"info2.values before merge: {info2.values}")

result = info1.merge(info2)

print(f"\nAfter merge:")
print(f"result is info1: {result is info1}")
print(f"Same object (aliasing): {result.values['libraries'] is info2.values['libraries']}")

# Test the aliasing issue
print("\nAdding 'lib3' to result.values['libraries']...")
result.values['libraries'].append('lib3')

print(f"info2 libraries: {info2.values['libraries']}")
print(f"result libraries: {result.values['libraries']}")

if 'lib3' in info2.values['libraries']:
    print("❌ BUG CONFIRMED: Modifying merged list affected source list (aliasing)")
else:
    print("✓ OK: Lists are independent")

# Test when key already exists
print("\n" + "=" * 60)
print("2. Testing when key already exists in self:")
info3 = DistutilsInfo()
info3.values['libraries'] = ['existing1']
info4 = DistutilsInfo()
info4.values['libraries'] = ['new1', 'new2']

print(f"info3.values before merge: {info3.values}")
print(f"info4.values before merge: {info4.values}")

info3.merge(info4)
print(f"\nAfter merge:")
print(f"info3 libraries: {info3.values['libraries']}")
print(f"info4 libraries: {info4.values['libraries']}")

print(f"Same object (aliasing): {info3.values['libraries'] is info4.values['libraries']}")

info3.values['libraries'].append('test')
print(f"\nAfter adding 'test' to info3:")
print(f"info3 libraries: {info3.values['libraries']}")
print(f"info4 libraries: {info4.values['libraries']}")

if 'test' in info4.values['libraries']:
    print("❌ BUG: Modifying merged list affected source list (aliasing)")
else:
    print("✓ OK: Lists are independent when key exists")

# Test with other transitive_list types
print("\n" + "=" * 60)
print("3. Testing other transitive_list keys:")
for key in ['library_dirs', 'runtime_library_dirs', 'include_dirs',
            'extra_compile_args', 'extra_link_args', 'depends']:
    info5 = DistutilsInfo()
    info6 = DistutilsInfo()
    info6.values[key] = ['value1', 'value2']

    info5.merge(info6)
    is_alias = info5.values[key] is info6.values[key]

    print(f"{key}: aliasing = {is_alias}")
    if is_alias:
        print(f"  ❌ BUG: {key} lists are aliased")