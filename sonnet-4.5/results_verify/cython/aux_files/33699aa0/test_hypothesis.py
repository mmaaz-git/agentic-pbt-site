import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given
import hypothesis.strategies as st
from Cython.Tempita._looper import looper

@given(st.lists(st.integers(), min_size=2, max_size=20))
def test_looper_odd_even_type_consistency(seq):
    results = list(looper(seq))
    for loop, item in results:
        assert isinstance(loop.odd, bool), f"odd should return bool, got {type(loop.odd)}"
        assert isinstance(loop.even, bool), f"even should return bool, got {type(loop.even)}"

# Test with a simple example
def test_manual():
    seq = [0, 0]
    results = list(looper(seq))
    for loop, item in results:
        assert isinstance(loop.odd, bool), f"odd should return bool, got {type(loop.odd)}"
        assert isinstance(loop.even, bool), f"even should return bool, got {type(loop.even)}"

try:
    test_manual()
    print("Test passed for [0, 0]")
except AssertionError as e:
    print(f"Test failed: {e}")