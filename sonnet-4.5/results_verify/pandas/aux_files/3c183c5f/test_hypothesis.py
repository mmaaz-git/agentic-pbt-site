from hypothesis import given, strategies as st, assume
from pandas.core.computation.scope import Scope


@given(
    st.text(min_size=1, max_size=10),
    st.text(min_size=1, max_size=10),
    st.integers(),
    st.integers()
)
def test_swapkey_removes_old_key(old_key, new_key, old_value, new_value):
    """
    Property: After swapkey(old_key, new_key, new_value):
    - new_key should exist with new_value
    - old_key should NOT exist (it should be removed)
    """
    assume(old_key != new_key)

    scope = Scope(level=0, global_dict={old_key: old_value})

    assert old_key in scope.scope
    assert scope.scope[old_key] == old_value

    scope.swapkey(old_key, new_key, new_value)

    assert new_key in scope.scope
    assert scope.scope[new_key] == new_value

    assert old_key not in scope.scope

# Run the test
if __name__ == "__main__":
    test_swapkey_removes_old_key()
    print("Test passed!")