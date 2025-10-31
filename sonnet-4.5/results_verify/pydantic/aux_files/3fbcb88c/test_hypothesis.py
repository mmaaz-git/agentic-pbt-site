#!/usr/bin/env python3
"""Run the hypothesis test from the bug report"""

import json
from hypothesis import given, strategies as st
from pydantic.deprecated.parse import load_str_bytes

@given(st.dictionaries(st.text(min_size=1), st.integers()))
def test_load_str_bytes_with_charset_parameter(data_dict):
    json_str = json.dumps(data_dict)

    result1 = load_str_bytes(json_str, content_type='application/json')
    assert result1 == data_dict

    result2 = load_str_bytes(json_str, content_type='application/json; charset=utf-8')
    assert result2 == data_dict, "Content-Type with charset parameter should work"

if __name__ == '__main__':
    print("Running hypothesis test...")
    test_load_str_bytes_with_charset_parameter()