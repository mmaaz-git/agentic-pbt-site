import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from hypothesis import given, strategies as st, settings
from django.db.models import Q


def q_objects():
    return st.builds(Q, x=st.integers()) | st.builds(Q, name=st.text())


@given(q_objects())
@settings(max_examples=500)
def test_q_and_idempotent(q):
    assert (q & q) == q


@given(q_objects())
@settings(max_examples=500)
def test_q_or_idempotent(q):
    assert (q | q) == q

if __name__ == "__main__":
    test_q_and_idempotent()
    test_q_or_idempotent()