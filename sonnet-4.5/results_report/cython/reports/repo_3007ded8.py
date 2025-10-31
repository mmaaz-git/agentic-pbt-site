import io
from Cython.Plex import *

lexicon = Lexicon([
    (Str('start'), Begin('state2')),
    State('state2', [(Str('x'), 'STATE2_X')])
])

scanner = Scanner(lexicon, io.StringIO('start'))

try:
    token, text = scanner.read()
    print(f'Success: {token!r}')
except AttributeError as e:
    print(f'AttributeError: {e}')