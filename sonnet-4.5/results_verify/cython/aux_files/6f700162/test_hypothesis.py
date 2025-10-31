import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, example
from Cython.TestUtils import _parse_pattern

@given(st.text(min_size=1, max_size=50))
@example("/start")
@example("/")
@example(":/end")
@example(":/")
def test_parse_pattern_should_not_crash(pattern):
    try:
        start, end, parsed = _parse_pattern(pattern)
        assert isinstance(start, (str, type(None)))
        assert isinstance(end, (str, type(None)))
        assert isinstance(parsed, str)
        print(f"✓ Pattern {repr(pattern)} -> start={repr(start)}, end={repr(end)}, parsed={repr(parsed)}")
    except Exception as e:
        print(f"✗ Pattern {repr(pattern)} failed with {type(e).__name__}: {e}")
        raise

if __name__ == "__main__":
    test_parse_pattern_should_not_crash()