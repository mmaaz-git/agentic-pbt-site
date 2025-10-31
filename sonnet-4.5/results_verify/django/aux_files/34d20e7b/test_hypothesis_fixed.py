#!/usr/bin/env python3
"""Run the hypothesis test from the bug report with health check suppressed"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from hypothesis import given, strategies as st, settings, assume, HealthCheck
from django.db import models
from django.db.migrations.operations import CreateModel, RenameField

# Create a more targeted strategy
def field_name_pair():
    """Generate pairs of field names that differ only in case"""
    base = st.text(min_size=1, max_size=10, alphabet='abcdefghijklmnopqrstuvwxyz')

    @st.composite
    def make_pair(draw):
        base_name = draw(base)
        # Create a variant with different case
        variant = ''.join(c.upper() if i % 2 == 0 else c for i, c in enumerate(base_name))
        new_name = draw(st.text(min_size=1, max_size=10, alphabet='abcdefghijklmnopqrstuvwxyz'))
        # Make sure new_name is different
        assume(new_name.lower() != base_name.lower())
        return base_name, variant, new_name

    return make_pair()

@given(field_name_pair())
@settings(max_examples=50, suppress_health_check=[HealthCheck.filter_too_much])
def test_rename_field_case_insensitive_in_constraints(names):
    old_field, constraint_field_variant, new_field = names

    # Only test if they differ in case but match when lowercased
    if constraint_field_variant.lower() != old_field.lower():
        return  # Skip this test case

    if constraint_field_variant == old_field:
        return  # Skip if they're identical (no case difference)

    create_op = CreateModel(
        name='MyModel',
        fields=[
            ('id', models.AutoField(primary_key=True)),
            ('other', models.CharField(max_length=100)),
            (old_field, models.CharField(max_length=100))
        ],
        options={'unique_together': {('other', constraint_field_variant)}}
    )

    rename_op = RenameField(
        model_name='MyModel',
        old_name=old_field,
        new_name=new_field
    )

    result = create_op.reduce(rename_op, app_label='test_app')
    unique_together = result[0].options.get('unique_together')

    for tup in unique_together:
        for f in tup:
            if f.lower() == old_field.lower():
                # This field should have been renamed
                assert f == new_field, \
                    f"Field not renamed: {f} should be {new_field} (old_field={old_field}, constraint_field={constraint_field_variant})"

print("Running hypothesis test...")
failures = 0
try:
    test_rename_field_case_insensitive_in_constraints()
    print("All tests passed!")
except AssertionError as e:
    print(f"Test failed: {e}")
    failures += 1
except Exception as e:
    print(f"Error running tests: {e}")
    failures += 1

# Also test the specific case from the bug report
print("\nTesting specific failing input from bug report...")
old_field = 'myField'
constraint_field_variant = 'MyField'
new_field = 'renamedField'

create_op = CreateModel(
    name='MyModel',
    fields=[
        ('id', models.AutoField(primary_key=True)),
        ('other', models.CharField(max_length=100)),
        (old_field, models.CharField(max_length=100))
    ],
    options={'unique_together': {('other', constraint_field_variant)}}
)

rename_op = RenameField(
    model_name='MyModel',
    old_name=old_field,
    new_name=new_field
)

result = create_op.reduce(rename_op, app_label='test_app')
unique_together = result[0].options.get('unique_together')

print(f"Input: old_field='{old_field}', constraint_field_variant='{constraint_field_variant}', new_field='{new_field}'")
print(f"Result: {unique_together}")
print(f"Expected: {{('other', '{new_field}')}}")

failed = False
for tup in unique_together:
    for f in tup:
        if f.lower() == old_field.lower() and f != new_field:
            print(f"FAILED: Field not renamed: {f} should be {new_field}")
            failed = True
            failures += 1

if not failed and failures == 0:
    print("SUCCESS: All tests passed")
else:
    print(f"FAILURE: {failures} test(s) failed")