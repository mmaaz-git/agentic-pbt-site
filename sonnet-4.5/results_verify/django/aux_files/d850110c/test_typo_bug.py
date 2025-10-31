#!/usr/bin/env python3
"""Test the typo bug in Django's Oracle backend"""

import re


def date_extract_sql_oracle(lookup_type, sql, params):
    """Simplified version of the Oracle date_extract_sql method"""
    _extract_format_re = re.compile(r"[A-Z_]+")

    extract_sql = f"TO_CHAR({sql}, %s)"
    extract_param = None
    if lookup_type == "week_day":
        extract_param = "D"
    elif lookup_type == "iso_week_day":
        extract_sql = f"TO_CHAR({sql} - 1, %s)"
        extract_param = "D"
    elif lookup_type == "week":
        extract_param = "IW"
    elif lookup_type == "quarter":
        extract_param = "Q"
    elif lookup_type == "iso_year":
        extract_param = "IYYY"
    else:
        lookup_type = lookup_type.upper()
        if not _extract_format_re.fullmatch(lookup_type):
            raise ValueError(f"Invalid loookup type: {lookup_type!r}")
        return f"EXTRACT({lookup_type} FROM {sql})", params
    return extract_sql, (*params, extract_param)


# Test 1: Reproduce the simple bug
print("Test 1: Reproducing the typo in error message")
try:
    date_extract_sql_oracle("invalid!type", "field", ())
except ValueError as e:
    print(f"Error message: {e}")
    if "loookup" in str(e):
        print("✓ Bug confirmed: Error message contains 'loookup' with 3 o's")
    else:
        print("✗ Bug not found: Error message does not contain the typo")


# Test 2: Run the hypothesis test
from hypothesis import given, strategies as st, settings as hyp_settings

@given(st.text(min_size=1, max_size=50))
@hyp_settings(max_examples=1000)
def test_date_extract_sql_validates_lookup_type(lookup_type):
    valid_regex = re.compile(r"[A-Z_]+")

    if lookup_type in ['week_day', 'iso_week_day', 'week', 'quarter', 'iso_year']:
        result_sql, result_params = date_extract_sql_oracle(lookup_type, 'field', ())
        assert result_sql is not None
        assert isinstance(result_params, tuple)
    elif valid_regex.fullmatch(lookup_type.upper()):
        result_sql, result_params = date_extract_sql_oracle(lookup_type, 'field', ())
        assert result_sql is not None
        assert isinstance(result_params, tuple)
    else:
        try:
            date_extract_sql_oracle(lookup_type, 'field', ())
            assert False, f"Expected ValueError for invalid lookup_type: {lookup_type!r}"
        except ValueError as e:
            error_msg = str(e)
            # Check that the error message contains both "Invalid" and "type"
            assert 'Invalid' in error_msg and 'type' in error_msg
            # Also check for the typo
            if 'loookup' in error_msg:
                pass  # Typo confirmed
            elif 'lookup' in error_msg:
                print(f"Note: Error message has correct spelling 'lookup' for input: {lookup_type!r}")

print("\nTest 2: Running hypothesis tests...")
test_date_extract_sql_validates_lookup_type()
print("✓ All hypothesis tests passed")