import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import django
from django.conf import settings as django_settings

if not django_settings.configured:
    django_settings.configure(
        DEBUG=True,
        SECRET_KEY='test-secret-key',
        DEFAULT_CHARSET='utf-8',
    )
    django.setup()

from hypothesis import given, strategies as st, settings
from django.http import QueryDict


@given(
    st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=20),
    st.text(max_size=30)
)
@settings(max_examples=500)
def test_querydict_pop_getitem_consistency(key, value):
    qd = QueryDict(mutable=True)
    qd[key] = value

    retrieved_via_getitem = qd[key]
    type_via_getitem = type(retrieved_via_getitem)

    qd2 = QueryDict(mutable=True)
    qd2[key] = value
    popped = qd2.pop(key)
    type_via_pop = type(popped)

    assert type_via_getitem == type_via_pop, \
        f"qd[key] and qd.pop(key) should return the same type.\n" \
        f"qd[{key!r}] returned {type_via_getitem.__name__}: {retrieved_via_getitem!r}\n" \
        f"qd.pop({key!r}) returned {type_via_pop.__name__}: {popped!r}"


if __name__ == "__main__":
    test_querydict_pop_getitem_consistency()