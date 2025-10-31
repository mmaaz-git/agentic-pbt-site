from hypothesis import given, strategies as st
import attr

@given(st.integers(), st.text(), st.text())
def test_eq_false_hash_true_contract_violation(val, text1, text2):
    from hypothesis import assume
    assume(text1 != text2)

    @attr.define(hash=True)
    class TestClass:
        x: int
        weird: str = attr.field(eq=False, hash=True)

    instance1 = TestClass(x=val, weird=text1)
    instance2 = TestClass(x=val, weird=text2)

    are_equal = (instance1 == instance2)
    if are_equal:
        assert hash(instance1) == hash(instance2), \
            f"Hash/equality contract violated: equal objects have different hashes. " \
            f"instance1={instance1}, instance2={instance2}, " \
            f"hash1={hash(instance1)}, hash2={hash(instance2)}"

# Run the test
if __name__ == "__main__":
    print("Running property-based test...")
    try:
        test_eq_false_hash_true_contract_violation()
        print("Test completed - no violations found")
    except AssertionError as e:
        print(f"Test failed with AssertionError: {e}")
    except Exception as e:
        print(f"Test failed with unexpected error: {e}")