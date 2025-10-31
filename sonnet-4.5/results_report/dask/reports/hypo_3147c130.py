from hypothesis import given, strategies as st
from django.utils import tree
from django.db.models.sql.where import AND, OR, XOR
import copy as python_copy


@given(st.sampled_from([AND, OR, XOR]), st.lists(st.integers(), min_size=1))
def test_copy_creates_independent_children_list(connector, children):
    node = tree.Node(children=children[:], connector=connector)
    copied = python_copy.copy(node)

    node.children.append(999)

    assert 999 not in copied.children

# Run the test
if __name__ == "__main__":
    test_copy_creates_independent_children_list()