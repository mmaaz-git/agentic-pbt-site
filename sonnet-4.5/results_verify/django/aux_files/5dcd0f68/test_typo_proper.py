#!/usr/bin/env python3
import os
import django
import sys

# Configure Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')
django.setup()

from django.db.backends.mysql.operations import DatabaseOperations
from unittest.mock import Mock

ops = DatabaseOperations(connection=Mock())

try:
    ops.date_extract_sql("invalid$type", "sql", [])
except ValueError as e:
    print(f"Error message: {e}")
    if "loookup" in str(e):
        print("BUG CONFIRMED: Typo found - 'loookup' instead of 'lookup'")
    else:
        print("BUG NOT FOUND: Correct spelling used")