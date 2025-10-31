import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from hypothesis import given, strategies as st
from django.db.models.functions import Collate
from django.db.models.expressions import Value


@given(st.text(alphabet=st.characters(whitelist_categories=['Lu', 'Ll', 'Nd']), min_size=1).map(lambda x: x + '\n'))
def test_collate_should_reject_trailing_newline(collation_with_newline):
    try:
        Collate(Value("test"), collation_with_newline)
        assert False, f"Collate should reject {repr(collation_with_newline)} but it didn't"
    except ValueError:
        pass

if __name__ == "__main__":
    test_collate_should_reject_trailing_newline()