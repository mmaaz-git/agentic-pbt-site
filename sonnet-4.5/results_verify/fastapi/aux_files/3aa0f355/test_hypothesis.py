import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/attrs_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from attrs import validators
import operator

@given(st.sampled_from([validators.lt, validators.le, validators.ge, validators.gt]))
def test_validator_docstring_matches_implementation(validator_func):
    validator = validator_func(0)
    docstring = validator_func.__doc__
    actual_op = validator.compare_func

    if 'operator.lt' in docstring:
        assert actual_op == operator.lt
    elif 'operator.le' in docstring:
        assert actual_op == operator.le
    elif 'operator.ge' in docstring:
        assert actual_op == operator.ge
    elif 'operator.gt' in docstring:
        assert actual_op == operator.gt

# Run the test
test_validator_docstring_matches_implementation()