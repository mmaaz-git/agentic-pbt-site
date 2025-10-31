import io
from hypothesis import given, strategies as st, settings
from Cython.Plex import *

@given(st.text(alphabet='abc', min_size=1, max_size=3))
@settings(max_examples=200)
def test_begin_action_changes_state(trigger_pattern):
    state1_pattern = 'x'
    state2_pattern = 'y'

    lexicon = Lexicon([
        (Str(trigger_pattern), Begin('state2')),
        State('state2', [
            (Str(state2_pattern), 'STATE2_TOKEN')
        ])
    ])

    scanner = Scanner(lexicon, io.StringIO(trigger_pattern + state2_pattern))

    token1, text1 = scanner.read()
    token2, text2 = scanner.read()

if __name__ == '__main__':
    test_begin_action_changes_state()