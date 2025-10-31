#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')

import django
from django.conf import settings
from django.db import connection

# Configure Django with a minimal in-memory database
settings.configure(
    DEBUG=True,
    DATABASES={
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': ':memory:',
        }
    },
    INSTALLED_APPS=[
        'django.contrib.contenttypes',
        'django.contrib.auth',
    ],
)
django.setup()

from django.db import models
from django.db.models.functions import Substr, Left
from django.db.models.expressions import Value

# Create a test model
class TestModel(models.Model):
    text = models.CharField(max_length=100)

    class Meta:
        app_label = 'test'

# Create the table
from django.db import connection
with connection.schema_editor() as schema_editor:
    schema_editor.create_model(TestModel)

# Insert test data
TestModel.objects.create(text="hello world")

print("Testing SQL generation and execution with invalid length values")
print("="*60)

# Test Substr with negative length
print("\n1. Testing Substr with negative length:")
try:
    # This should generate SQL like: SUBSTR(text, 1, -5)
    qs = TestModel.objects.annotate(
        result=Substr('text', 1, -5)
    )
    print(f"Generated SQL: {qs.query}")
    results = list(qs.values('result'))
    print(f"Query executed successfully. Result: {results}")
except Exception as e:
    print(f"Query failed with error: {e}")

# Test Substr with zero length
print("\n2. Testing Substr with zero length:")
try:
    qs = TestModel.objects.annotate(
        result=Substr('text', 1, 0)
    )
    print(f"Generated SQL: {qs.query}")
    results = list(qs.values('result'))
    print(f"Query executed successfully. Result: {results}")
except Exception as e:
    print(f"Query failed with error: {e}")

# Test standard SQL behavior with SUBSTR
print("\n3. Testing raw SQL SUBSTR with negative/zero length:")
with connection.cursor() as cursor:
    # Test negative length
    try:
        cursor.execute("SELECT SUBSTR('hello world', 1, -5)")
        result = cursor.fetchone()
        print(f"SUBSTR('hello world', 1, -5) = {result}")
    except Exception as e:
        print(f"SUBSTR with negative length failed: {e}")

    # Test zero length
    try:
        cursor.execute("SELECT SUBSTR('hello world', 1, 0)")
        result = cursor.fetchone()
        print(f"SUBSTR('hello world', 1, 0) = {result}")
    except Exception as e:
        print(f"SUBSTR with zero length failed: {e}")

print("\n" + "="*60)
print("Checking what different databases do with negative/zero length:")
print("="*60)

# SQLite behavior
print("\nSQLite behavior (current database):")
with connection.cursor() as cursor:
    test_cases = [
        ("SUBSTR('hello', 1, 0)", "zero length"),
        ("SUBSTR('hello', 1, -1)", "negative length"),
        ("SUBSTR('hello', 1, -5)", "large negative length"),
    ]
    for sql, desc in test_cases:
        try:
            cursor.execute(f"SELECT {sql}")
            result = cursor.fetchone()[0]
            print(f"  {sql} = '{result}' (length={len(result) if result else 0})")
        except Exception as e:
            print(f"  {sql} - Error: {e}")