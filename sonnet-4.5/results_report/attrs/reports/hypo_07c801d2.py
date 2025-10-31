from hypothesis import given, strategies as st, assume
import attrs

@given(st.integers(), st.integers(), st.integers())
def test_hash_equality_contract_with_eq_false_hash_true(shared, val1, val2):
    """Objects that are equal MUST have equal hashes (Python requirement)"""
    @attrs.define(hash=True)
    class Data:
        shared: int
        excluded: int = attrs.field(eq=False, hash=True)

    obj1 = Data(shared, val1)
    obj2 = Data(shared, val2)

    assume(val1 != val2)

    assert obj1 == obj2
    assert hash(obj1) == hash(obj2)

if __name__ == "__main__":
    test_hash_equality_contract_with_eq_false_hash_true()