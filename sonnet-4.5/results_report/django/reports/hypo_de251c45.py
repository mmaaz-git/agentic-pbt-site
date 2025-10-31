import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, assume, example
from django.core.cache.utils import make_template_fragment_key

@given(
    fragment_name=st.text(),
    list1=st.lists(st.text(), min_size=1, max_size=5),
    list2=st.lists(st.text(), min_size=1, max_size=5)
)
@example(fragment_name="fragment", list1=["a:", "b"], list2=["a", ":b"])
@example(fragment_name="test", list1=["x:y", "z"], list2=["x", "y:z"])
def test_different_inputs_should_produce_different_keys(fragment_name, list1, list2):
    assume(list1 != list2)

    key1 = make_template_fragment_key(fragment_name, list1)
    key2 = make_template_fragment_key(fragment_name, list2)

    assert key1 != key2, f"Cache key collision: {list1} and {list2} produce same key"

if __name__ == "__main__":
    test_different_inputs_should_produce_different_keys()