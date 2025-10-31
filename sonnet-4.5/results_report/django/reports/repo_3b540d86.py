#!/usr/bin/env python3
"""
Minimal reproduction of the IndexError bug in django.templatetags.i18n.GetLanguageInfoListNode.get_language_info
"""

import sys

# Add Django to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Configure Django settings
from django.conf import settings
settings.configure(
    INSTALLED_APPS=['django.contrib.contenttypes'],
    LANGUAGES=[('en', 'English'), ('es', 'Spanish')],
    USE_I18N=True,
    USE_TZ=False,
    SECRET_KEY='test-secret-key'
)

# Initialize Django
import django
django.setup()

# Import the class with the bug
from django.templatetags.i18n import GetLanguageInfoListNode

# Create an instance
node = GetLanguageInfoListNode(None, None)

# Test with empty string - This will crash with IndexError
print("Testing with empty string '':")
try:
    result = node.get_language_info('')
    print(f"Result: {result}")
except IndexError as e:
    print(f"IndexError: {e}")

# Test with empty tuple - This will also crash
print("\nTesting with empty tuple ():")
try:
    result = node.get_language_info(())
    print(f"Result: {result}")
except IndexError as e:
    print(f"IndexError: {e}")

# Test with empty list - This will also crash
print("\nTesting with empty list []:")
try:
    result = node.get_language_info([])
    print(f"Result: {result}")
except IndexError as e:
    print(f"IndexError: {e}")

# Test with valid input for comparison
print("\nTesting with valid input 'en':")
try:
    result = node.get_language_info('en')
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")