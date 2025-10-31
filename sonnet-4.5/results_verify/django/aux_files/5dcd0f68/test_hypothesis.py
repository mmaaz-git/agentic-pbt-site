#!/usr/bin/env python3
import os
import django

# Configure Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')
django.setup()

from hypothesis import given, strategies as st, settings as hyp_settings
from django.db.backends.mysql.operations import DatabaseOperations
from unittest.mock import Mock

@given(st.text(min_size=1, max_size=20))
@hyp_settings(max_examples=500)
def test_date_extract_sql_error_message_typo(lookup_type):
    ops = DatabaseOperations(connection=Mock())
    sql = "DATE_COLUMN"
    params = []

    try:
        ops.date_extract_sql(lookup_type, sql, params)
    except ValueError as e:
        error_msg = str(e)
        if "loookup" in error_msg:
            print(f"Typo found with input '{lookup_type}': {error_msg!r}")
            assert False, f"Typo found: {error_msg!r}"

print("Running Hypothesis test...")
test_date_extract_sql_error_message_typo()
print("Test completed")