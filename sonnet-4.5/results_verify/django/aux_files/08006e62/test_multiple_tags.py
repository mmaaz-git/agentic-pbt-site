#!/usr/bin/env python3
"""Test if registering with multiple tags in one call works"""

import sys
import os

# Add Django to Python path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Set up Django settings
os.environ['DJANGO_SETTINGS_MODULE'] = 'django.conf.global_settings'

import django
django.setup()

from django.core.checks.registry import CheckRegistry
from django.core.checks import Info

print("Test 1: Registering with multiple tags in one call")
registry1 = CheckRegistry()

def my_check1(app_configs=None, **kwargs):
    return [Info("My check")]

# Register with multiple tags in one call
registry1.register(my_check1, "tag1", "tag2")
print(f"Tags after registration: {my_check1.tags}")

available_tags1 = registry1.tags_available()
print(f"Available tags: {available_tags1}")

checks_tag1 = registry1.run_checks(tags=["tag1"])
checks_tag2 = registry1.run_checks(tags=["tag2"])

print(f"Checks with tag1: {len(checks_tag1)}")
print(f"Checks with tag2: {len(checks_tag2)}")

print("\n" + "="*50)
print("Test 2: The problematic case - registering twice with different tags")
registry2 = CheckRegistry()

def my_check2(app_configs=None, **kwargs):
    return [Info("My check")]

registry2.register(my_check2, "tag1")
print(f"After first registration - tags: {my_check2.tags}")

registry2.register(my_check2, "tag2")
print(f"After second registration - tags: {my_check2.tags}")

available_tags2 = registry2.tags_available()
print(f"Available tags: {available_tags2}")

checks2_tag1 = registry2.run_checks(tags=["tag1"])
checks2_tag2 = registry2.run_checks(tags=["tag2"])

print(f"Checks with tag1: {len(checks2_tag1)}")
print(f"Checks with tag2: {len(checks2_tag2)}")