import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from django.utils.http import is_same_domain

@given(host=st.text(
    alphabet=st.characters(whitelist_categories=('Ll', 'Lu', 'Nd'), whitelist_characters='.-'),
    min_size=1,
    max_size=50
))
@settings(max_examples=200)
def test_is_same_domain_exact_match(host):
    """Property: is_same_domain should return True for exact match"""
    result = is_same_domain(host, host)
    assert result == True, f"Exact match failed: is_same_domain({host!r}, {host!r}) = {result}"

if __name__ == "__main__":
    test_is_same_domain_exact_match()