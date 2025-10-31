import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from hypothesis import given, strategies as st, settings, assume
from django.db.models import Q


@st.composite
def q_objects(draw):
    field_name = draw(st.sampled_from(['name', 'age', 'city', 'email', 'status']))
    value = draw(st.one_of(
        st.text(min_size=1, max_size=20),
        st.integers(min_value=0, max_value=100),
        st.booleans()
    ))
    return Q(**{field_name: value})


@given(q_objects(), q_objects())
@settings(max_examples=200)
def test_q_and_commutativity(q1, q2):
    assume(q1 != q2)
    result1 = q1 & q2
    result2 = q2 & q1
    assert result1 == result2, f"Expected {result1} == {result2}"


@given(q_objects())
@settings(max_examples=200)
def test_q_and_idempotence(q):
    result = q & q
    assert result == q, f"Expected {result} == {q}"

if __name__ == "__main__":
    print("Testing commutativity...")
    try:
        test_q_and_commutativity()
        print("Commutativity test passed")
    except AssertionError as e:
        print(f"Commutativity test failed: {e}")

    print("\nTesting idempotence...")
    try:
        test_q_and_idempotence()
        print("Idempotence test passed")
    except AssertionError as e:
        print(f"Idempotence test failed: {e}")