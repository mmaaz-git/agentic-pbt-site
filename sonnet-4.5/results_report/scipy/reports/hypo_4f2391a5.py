#!/usr/bin/env python3
"""Hypothesis test for Django SQLite3 _sqlite_date_trunc bug"""

from hypothesis import given, settings
import hypothesis.strategies as st
import django.db.backends.sqlite3._functions as funcs

@given(st.sampled_from(['year', 'month', 'day', 'week', 'quarter']))
@settings(max_examples=100)
def test_date_trunc_idempotent(lookup_type):
    dt = "2023-06-15"
    conn_tzname = "UTC"
    truncated_once = funcs._sqlite_date_trunc(lookup_type, dt, None, conn_tzname)
    if truncated_once is not None:
        truncated_twice = funcs._sqlite_date_trunc(lookup_type, truncated_once, None, conn_tzname)
        assert truncated_once == truncated_twice

if __name__ == "__main__":
    # Run the test
    test_date_trunc_idempotent()