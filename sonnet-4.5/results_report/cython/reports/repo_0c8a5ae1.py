#!/usr/bin/env python3
"""
Minimal reproduction case for Cython.Plex.Regexps.RE.wrong_type bug
This demonstrates that wrong_type crashes with AttributeError in Python 3
instead of raising the intended PlexTypeError.
"""

from Cython.Plex import Seq, Str
from Cython.Plex.Errors import PlexTypeError

try:
    # This should raise PlexTypeError with a helpful message about wrong type
    # Instead it crashes with AttributeError about missing types.InstanceType
    seq = Seq(Str('a'), "not an RE")
except PlexTypeError as e:
    print(f"PlexTypeError (expected): {e}")
except AttributeError as e:
    print(f"AttributeError (unexpected): {e}")
    print("\nThis should have raised PlexTypeError with a helpful message.")
    print("Instead it crashes because types.InstanceType doesn't exist in Python 3.")