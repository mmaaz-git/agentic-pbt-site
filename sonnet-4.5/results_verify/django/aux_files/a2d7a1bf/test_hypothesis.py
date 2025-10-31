"""Run the Hypothesis property-based test from the bug report."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from hypothesis import given, settings, strategies as st
from django.db.models import Q


@st.composite
def q_objects(draw):
    """Generate Q objects with simple field lookups."""
    field_names = ['id', 'name', 'value', 'count']
    field = draw(st.sampled_from(field_names))
    value = draw(st.one_of(
        st.integers(),
        st.text(min_size=0, max_size=10),
        st.booleans()
    ))
    return Q(**{field: value})


@given(q_objects(), q_objects())
@settings(max_examples=100)
def test_q_and_commutative(q1, q2):
    """Q objects should satisfy commutativity for AND: q1 & q2 == q2 & q1."""
    result1 = q1 & q2
    result2 = q2 & q1
    if result1 != result2:
        print(f"Found failing case: q1={q1}, q2={q2}")
        print(f"  q1 & q2 = {result1}")
        print(f"  q2 & q1 = {result2}")
        assert False, f"AND not commutative: {result1} != {result2}"

# Run the test
print("Running Hypothesis test for Q object commutativity...")
try:
    test_q_and_commutative()
    print("Test passed with 100 examples!")
except AssertionError as e:
    print(f"Test failed: {e}")
except Exception as e:
    print(f"Error running test: {e}")