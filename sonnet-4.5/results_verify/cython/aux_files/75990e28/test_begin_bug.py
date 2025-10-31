#!/usr/bin/env python3
"""Test to reproduce the Begin action bug"""

import io
import sys
from Cython.Plex import *

# Test 1: Basic reproduction
print("=" * 60)
print("Test 1: Basic Begin action usage")
print("=" * 60)

lexicon = Lexicon([
    (Str('start'), Begin('state2')),
    State('state2', [(Str('x'), 'STATE2_X')])
])

scanner = Scanner(lexicon, io.StringIO('start'))

try:
    token, text = scanner.read()
    print(f'Success: token={token!r}, text={text!r}')
except AttributeError as e:
    print(f'AttributeError: {e}')
except Exception as e:
    print(f'Unexpected error: {type(e).__name__}: {e}')

# Test 2: Property-based test from bug report
print("\n" + "=" * 60)
print("Test 2: Property-based test")
print("=" * 60)

try:
    from hypothesis import given, strategies as st, settings

    @given(st.text(alphabet='abc', min_size=1, max_size=3))
    @settings(max_examples=5)
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
        print(f"  Pattern '{trigger_pattern}': token1={token1}, text1={text1}, token2={token2}, text2={text2}")

    test_begin_action_changes_state()
    print("Property test passed!")

except ImportError:
    print("Hypothesis not installed, skipping property test")
except Exception as e:
    print(f'Property test failed: {type(e).__name__}: {e}')

# Test 3: Check if Scanner has begin() method
print("\n" + "=" * 60)
print("Test 3: Scanner method inspection")
print("=" * 60)

scanner_methods = [attr for attr in dir(Scanner) if not attr.startswith('_')]
print("Public Scanner methods:", sorted(scanner_methods))
print(f"Has 'begin' method: {hasattr(Scanner, 'begin')}")
print(f"Has 'produce' method: {hasattr(Scanner, 'produce')}")

# Test 4: Direct method call test
print("\n" + "=" * 60)
print("Test 4: Direct method call test")
print("=" * 60)

lexicon = Lexicon([
    (Str('x'), 'X_TOKEN'),
    State('state2', [(Str('y'), 'Y_TOKEN')])
])
scanner = Scanner(lexicon, io.StringIO('xy'))

try:
    # Try calling begin directly
    scanner.begin('state2')
    print("Successfully called scanner.begin('state2')")

    # Read a token
    token, text = scanner.read()
    print(f"After begin('state2'), read: token={token}, text={text}")
except AttributeError as e:
    print(f"AttributeError when calling begin(): {e}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")