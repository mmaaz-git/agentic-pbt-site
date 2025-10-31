#!/usr/bin/env python3
"""Check how Django handles field names and database columns"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import django
from django.conf import settings

# Configure Django minimally
settings.configure(
    DEBUG=True,
    DATABASES={
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': ':memory:',
        }
    },
    INSTALLED_APPS=['django.contrib.contenttypes', 'django.contrib.auth'],
    USE_TZ=True,
)

django.setup()

from django.db import models

# Test how Django handles field names
class TestModel(models.Model):
    myField = models.CharField(max_length=100)
    MyOtherField = models.CharField(max_length=100)

    class Meta:
        app_label = 'test'

# Check field names and attributes
print("Field name mapping:")
for field in TestModel._meta.get_fields():
    if hasattr(field, 'column'):
        print(f'  Field name: {field.name}, DB column: {field.column}, attname: {field.attname}')

print(f"\nField lookup by name:")
try:
    field1 = TestModel._meta.get_field('myField')
    print(f"  get_field('myField'): {field1.name}")
except Exception as e:
    print(f"  get_field('myField'): Error - {e}")

try:
    field2 = TestModel._meta.get_field('MyField')  # Different case
    print(f"  get_field('MyField'): {field2.name}")
except Exception as e:
    print(f"  get_field('MyField'): Error - {e}")

try:
    field3 = TestModel._meta.get_field('myotherfield')  # All lowercase
    print(f"  get_field('myotherfield'): {field3.name}")
except Exception as e:
    print(f"  get_field('myotherfield'): Error - {e}")

print(f"\nTesting unique_together behavior:")

# Test with unique_together
class TestModel2(models.Model):
    myField = models.CharField(max_length=100)
    other = models.CharField(max_length=100)

    class Meta:
        app_label = 'test2'
        unique_together = [['other', 'MyField']]  # Note different case

print(f"  Model defined with field 'myField' and unique_together using 'MyField'")
print(f"  unique_together value: {TestModel2._meta.unique_together}")

# Test field references
from django.db.migrations.utils import field_references

print(f"\nField reference testing:")
references1 = field_references('myField', 'myField', TestModel2)
print(f"  field_references('myField', 'myField', TestModel2): {references1}")

references2 = field_references('MyField', 'myField', TestModel2)
print(f"  field_references('MyField', 'myField', TestModel2): {references2}")

references3 = field_references('myField', 'MyField', TestModel2)
print(f"  field_references('myField', 'MyField', TestModel2): {references3}")