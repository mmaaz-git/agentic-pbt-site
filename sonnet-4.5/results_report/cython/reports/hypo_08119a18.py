#!/usr/bin/env python3
"""Property-based test for Cython.Tempita sub() __name parameter leak"""

import sys
import string
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, settings, strategies as st
from Cython.Tempita import sub

@given(st.text(alphabet=string.ascii_letters, min_size=1, max_size=20))
@settings(max_examples=100)
def test_sub_name_parameter_isolation(template_name):
    """Test that __name parameter does not leak into template namespace"""
    content = "{{__name}}"

    result = sub(content, __name=template_name)

    assert result == '', f"__name should not be accessible in template namespace, but got: {repr(result)}"

if __name__ == "__main__":
    test_sub_name_parameter_isolation()