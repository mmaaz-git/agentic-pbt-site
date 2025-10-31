#!/usr/bin/env python3
"""Test the reported bug in Django's CreateModel.reduce() method"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

# First, test the reproduction case
print("=" * 60)
print("Testing the reported bug reproduction case")
print("=" * 60)

from django.db import models
from django.db.migrations.operations import CreateModel, RenameField

# Test case 1: Exact reproduction from the bug report
create_op = CreateModel(
    name='MyModel',
    fields=[
        ('id', models.AutoField(primary_key=True)),
        ('other', models.CharField(max_length=100)),
        ('myField', models.CharField(max_length=100))  # lowercase 'my'
    ],
    options={'unique_together': {('other', 'MyField')}}  # uppercase 'My'
)

rename_op = RenameField(
    model_name='MyModel',
    old_name='myField',  # matches field definition exactly
    new_name='renamedField'
)

result = create_op.reduce(rename_op, app_label='test_app')

print(f"Original field in definition: 'myField'")
print(f"Field in unique_together: 'MyField'")
print(f"After rename, unique_together is: {result[0].options.get('unique_together')}")
print(f"Expected: {{('other', 'renamedField')}}")
print(f"Bug exists: {'MyField' in str(result[0].options.get('unique_together'))}")
print()

# Test case 2: Case-matching scenario (control test)
print("=" * 60)
print("Control test: same case in field definition and constraint")
print("=" * 60)

create_op2 = CreateModel(
    name='MyModel2',
    fields=[
        ('id', models.AutoField(primary_key=True)),
        ('other', models.CharField(max_length=100)),
        ('myField', models.CharField(max_length=100))
    ],
    options={'unique_together': {('other', 'myField')}}  # same case as definition
)

rename_op2 = RenameField(
    model_name='MyModel2',
    old_name='myField',
    new_name='renamedField'
)

result2 = create_op2.reduce(rename_op2, app_label='test_app')
print(f"Original field in definition: 'myField'")
print(f"Field in unique_together: 'myField'")
print(f"After rename, unique_together is: {result2[0].options.get('unique_together')}")
print(f"Expected: {{('other', 'renamedField')}}")
print(f"Works correctly: {'renamedField' in str(result2[0].options.get('unique_together'))}")
print()

# Test case 3: Test with index_together
print("=" * 60)
print("Testing with index_together")
print("=" * 60)

create_op3 = CreateModel(
    name='MyModel3',
    fields=[
        ('id', models.AutoField(primary_key=True)),
        ('other', models.CharField(max_length=100)),
        ('myField', models.CharField(max_length=100))
    ],
    options={'index_together': {('other', 'MyField')}}  # uppercase 'My' in index
)

rename_op3 = RenameField(
    model_name='MyModel3',
    old_name='myField',
    new_name='renamedField'
)

result3 = create_op3.reduce(rename_op3, app_label='test_app')
print(f"Original field in definition: 'myField'")
print(f"Field in index_together: 'MyField'")
print(f"After rename, index_together is: {result3[0].options.get('index_together')}")
print(f"Expected: {{('other', 'renamedField')}}")
print(f"Bug exists: {'MyField' in str(result3[0].options.get('index_together'))}")
print()

# Test case 4: Test with order_with_respect_to
print("=" * 60)
print("Testing with order_with_respect_to")
print("=" * 60)

create_op4 = CreateModel(
    name='MyModel4',
    fields=[
        ('id', models.AutoField(primary_key=True)),
        ('myField', models.CharField(max_length=100))
    ],
    options={'order_with_respect_to': 'MyField'}  # uppercase 'My'
)

rename_op4 = RenameField(
    model_name='MyModel4',
    old_name='myField',
    new_name='renamedField'
)

result4 = create_op4.reduce(rename_op4, app_label='test_app')
print(f"Original field in definition: 'myField'")
print(f"Field in order_with_respect_to: 'MyField'")
print(f"After rename, order_with_respect_to is: {result4[0].options.get('order_with_respect_to')}")
print(f"Expected: 'renamedField'")
print(f"Bug exists: {result4[0].options.get('order_with_respect_to') == 'MyField'}")