#!/usr/bin/env python3
"""
Hypothesis test demonstrating the Cython debugger CyBreak.complete bug.
This test shows that already-typed function names appear in completion suggestions.
"""

from hypothesis import given, strategies as st, settings, example


def complete_unqualified_logic(text, word, all_names):
    """
    Extracted logic from CyBreak.complete method for unqualified name completion.
    This is the actual code from lines 957-959 of libcython.py.
    """
    word = word or ""
    seen = set(text[:-len(word)].split())
    return [n for n in all_names if n.startswith(word) and n not in seen]


@given(st.text(min_size=1, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))))
@example("spam")  # Explicit example from the bug report
@settings(max_examples=10)
def test_complete_with_empty_word(funcname):
    """
    Test that demonstrates the bug: when word is empty, already-typed
    function names incorrectly appear in completion suggestions.
    """
    word = ""
    text = f"cy break {funcname} "  # User typed the function name and a space
    all_names = [funcname, "other_func", "another_func"]

    result = complete_unqualified_logic(text, word, all_names)

    # This assertion PASSES, confirming the bug exists
    # The function name should NOT be in the result since it's already typed
    assert funcname in result, f"Bug confirmed: '{funcname}' should not be in suggestions but is present"

    print(f"✓ Bug reproduced with funcname='{funcname}'")
    print(f"  text = {repr(text)}")
    print(f"  word = {repr(word)}")
    print(f"  result = {result}")
    print(f"  '{funcname}' incorrectly appears in completion suggestions")


if __name__ == "__main__":
    print("Running hypothesis test for CyBreak.complete bug...")
    print("=" * 60)
    test_complete_with_empty_word()
    print("=" * 60)
    print("All tests passed, confirming the bug exists.")
    print("The bug: already-typed function names appear in completions when word=''")