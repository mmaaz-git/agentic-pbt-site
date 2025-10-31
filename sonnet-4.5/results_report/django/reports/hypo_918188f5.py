from hypothesis import given, settings, strategies as st
from django.dispatch import Signal

@settings(max_examples=500)
@given(st.booleans())
def test_has_listeners_consistency(use_caching):
    signal = Signal(use_caching=use_caching)

    assert not signal.has_listeners()

    def receiver(**kwargs):
        pass

    signal.connect(receiver, weak=False)
    assert signal.has_listeners()

    signal.disconnect(receiver)
    assert not signal.has_listeners()

# Run the test
if __name__ == "__main__":
    test_has_listeners_consistency()