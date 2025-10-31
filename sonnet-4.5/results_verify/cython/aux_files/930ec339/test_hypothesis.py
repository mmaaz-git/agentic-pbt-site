#!/usr/bin/env python3
"""Run the property-based test from the bug report"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

import string
from hypothesis import given
import hypothesis.strategies as st
from Cython.Tempita import Template

@given(st.text(alphabet=string.ascii_letters + '_', min_size=1).filter(str.isidentifier),
       st.integers(min_value=0, max_value=100),
       st.integers(min_value=101, max_value=200))
def test_substitute_args_override_namespace(var_name, namespace_value, substitute_value):
    content = f"{{{{{var_name}}}}}"
    template = Template(content, namespace={var_name: namespace_value})
    result = template.substitute({var_name: substitute_value})

    expected = str(substitute_value)
    actual = result

    if result != expected:
        print(f"FAIL: var_name={var_name}, namespace_value={namespace_value}, substitute_value={substitute_value}")
        print(f"  Expected: {expected}")
        print(f"  Got: {actual}")
        assert False, f"Expected {expected}, got {actual}"
    else:
        print(f"PASS: var_name={var_name}, namespace_value={namespace_value}, substitute_value={substitute_value}")

# Run the test
print("Running property-based test...")
test_substitute_args_override_namespace()