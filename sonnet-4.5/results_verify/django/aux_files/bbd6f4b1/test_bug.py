#!/usr/bin/env python3
"""Test script to reproduce the reported Django include() bug"""

# First test: Simple reproduction of the bug
print("=== Test 1: Simple reproduction ===")
from django.conf.urls import include
from django.core.exceptions import ImproperlyConfigured

patterns = []

try:
    result = include((patterns, ''), namespace='my_namespace')
    print(f"SUCCESS: include() returned: {result}")
    urlconf_module, app_name, ns = result
    print(f"  urlconf_module: {urlconf_module}")
    print(f"  app_name: {app_name!r}")
    print(f"  namespace: {ns}")
except ImproperlyConfigured as e:
    print(f"BUG CONFIRMED: ImproperlyConfigured raised: {e}")

print("\n=== Test 2: Control test with None app_name ===")
try:
    result = include((patterns, None), namespace='my_namespace')
    print(f"ERROR: This should have raised ImproperlyConfigured but returned: {result}")
except ImproperlyConfigured as e:
    print(f"EXPECTED: ImproperlyConfigured raised: {e}")

print("\n=== Test 3: Control test with non-empty app_name ===")
try:
    result = include((patterns, 'myapp'), namespace='my_namespace')
    urlconf_module, app_name, ns = result
    print(f"SUCCESS: include() with non-empty app_name works")
    print(f"  app_name: {app_name!r}")
    print(f"  namespace: {ns}")
except ImproperlyConfigured as e:
    print(f"UNEXPECTED ERROR: {e}")

print("\n=== Test 4: Empty string app_name without namespace ===")
try:
    result = include((patterns, ''))
    urlconf_module, app_name, ns = result
    print(f"SUCCESS: include() with empty app_name and no namespace works")
    print(f"  app_name: {app_name!r}")
    print(f"  namespace: {ns!r}")
except ImproperlyConfigured as e:
    print(f"UNEXPECTED ERROR: {e}")

print("\n=== Analysis of the bug ===")
print(f"Python truthiness check: not '' = {not ''}")
print(f"Python truthiness check: not None = {not None}")
print(f"Identity check: '' is None = {'' is None}")
print("\nThe bug occurs on line 42 of django/urls/conf.py:")
print("    if namespace and not app_name:")
print("This treats empty string ('') the same as None because both are falsy.")
print("The fix should be:")
print("    if namespace and app_name is None:")