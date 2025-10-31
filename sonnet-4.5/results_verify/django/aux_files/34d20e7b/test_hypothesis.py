#!/usr/bin/env python3
"""Run the hypothesis test from the bug report"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from hypothesis import given, strategies as st, settings, assume
from django.db import models
from django.db.migrations.operations import CreateModel, RenameField

field_name_strategy = st.text(min_size=1, max_size=30, alphabet=st.characters(
    whitelist_categories=('Lu', 'Ll'), min_codepoint=65, max_codepoint=122
))

@given(
    old_field=field_name_strategy,
    new_field=field_name_strategy,
    constraint_field_variant=field_name_strategy
)
@settings(max_examples=200)
def test_rename_field_case_insensitive_in_constraints(old_field, new_field, constraint_field_variant):
    assume(old_field.lower() != new_field.lower())
    assume(constraint_field_variant.lower() == old_field.lower())
    assume(constraint_field_variant != old_field)

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
            assert f.lower() != old_field.lower() or f.lower() == new_field.lower(), \
                f"Field not renamed: {f} should be {new_field} (old_field={old_field}, constraint_field={constraint_field_variant})"

print("Running hypothesis test with 200 examples...")
try:
    test_rename_field_case_insensitive_in_constraints()
    print("All tests passed!")
except AssertionError as e:
    print(f"Test failed: {e}")
except Exception as e:
    print(f"Error running tests: {e}")