#!/usr/bin/env python3
"""Hypothesis test for dask.utils.parse_bytes whitespace handling."""

from hypothesis import given, strategies as st, assume, settings, HealthCheck
import dask.utils
import pytest


@given(st.text(st.characters(whitelist_categories=('Zs', 'Cc')), min_size=1))
@settings(suppress_health_check=[HealthCheck.filter_too_much])
def test_parse_bytes_rejects_whitespace_only(s):
    assume(s.strip() == '')

    with pytest.raises(ValueError):
        dask.utils.parse_bytes(s)


if __name__ == "__main__":
    test_parse_bytes_rejects_whitespace_only()