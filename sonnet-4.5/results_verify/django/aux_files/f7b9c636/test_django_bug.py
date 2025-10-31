#!/usr/bin/env python3
import sys
import os

# Add Django environment to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

# Set up Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

import django
from django.conf import settings

# Configure minimal Django settings for SQLite
settings.configure(
    DATABASES={
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': ':memory:',
        }
    },
    DEBUG=True,
    INSTALLED_APPS=['django.contrib.contenttypes'],
)

django.setup()

from django.db import connection

print("Testing Django's _quote_params_for_last_executed_query with empty parameters...")

# Ensure connection is established
connection.ensure_connection()

# Test with empty tuple
params = ()
print(f"Parameters: {params}")

try:
    result = connection.ops._quote_params_for_last_executed_query(params)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test with empty list
params_list = []
print(f"\nTesting with list: {params_list}")

try:
    result = connection.ops._quote_params_for_last_executed_query(tuple(params_list))
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")