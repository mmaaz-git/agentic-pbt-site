#!/usr/bin/env python3
"""Test Django's actual logic for handling COMP_WORDS and COMP_CWORD."""

import os

def simulate_django_autocomplete():
    """Simulate Django's autocomplete logic from lines 303-309."""

    test_cases = [
        # (COMP_WORDS string, COMP_CWORD value, description)
        ("django-admin", 0, "COMP_CWORD=0 - cursor at command name itself"),
        ("django-admin", 1, "COMP_CWORD=1 - cursor after command, ready for subcommand"),
        ("django-admin migrate", 1, "COMP_CWORD=1 - cursor at 'migrate'"),
        ("django-admin migrate", 2, "COMP_CWORD=2 - cursor after 'migrate'"),
        ("django-admin migrate --database", 3, "COMP_CWORD=3 - cursor after '--database'"),
    ]

    for comp_words_str, comp_cword, description in test_cases:
        print(f"\nTest: {description}")
        print("-" * 50)
        print(f"COMP_WORDS = '{comp_words_str}'")
        print(f"COMP_CWORD = {comp_cword}")

        # Django's actual logic (lines 303-309)
        # Line 303: cwords = os.environ["COMP_WORDS"].split()[1:]
        cwords = comp_words_str.split()[1:]
        print(f"cwords (after split()[1:]) = {cwords}")

        # Line 304: cword = int(os.environ["COMP_CWORD"])
        # (we already have it as integer)

        # Lines 306-309:
        # try:
        #     curr = cwords[cword - 1]
        # except IndexError:
        #     curr = ""

        print(f"Attempting: curr = cwords[{comp_cword} - 1] = cwords[{comp_cword - 1}]")

        try:
            curr = cwords[comp_cword - 1]
            print(f"Result: curr = '{curr}'")
        except IndexError as e:
            curr = ""
            print(f"IndexError caught: {e}")
            print(f"Result: curr = ''")

        # Analysis
        print(f"\nAnalysis:")
        if comp_cword == 0:
            if len(cwords) > 0:
                print(f"⚠️  BUG: COMP_CWORD=0 with non-empty cwords")
                print(f"    Expected: curr = '' (cursor at command name)")
                print(f"    Actual: curr = '{curr}' (accessed via negative index)")
            else:
                print(f"✓ COMP_CWORD=0 with empty cwords handled correctly")
        else:
            print(f"✓ Normal case: COMP_CWORD={comp_cword}")

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Django's autocomplete() logic")
    print("=" * 60)
    simulate_django_autocomplete()

    print("\n" + "=" * 60)
    print("CONCLUSION:")
    print("=" * 60)
    print("""
The issue occurs when COMP_CWORD=0, which bash sets when the cursor is at the
command name itself (before any arguments). This is an edge case that could
happen during tab completion.

Django's code assumes that after split()[1:], the index (cword - 1) will either:
1. Be a valid positive index into cwords, OR
2. Raise an IndexError that gets caught

However, when COMP_CWORD=0:
- cword - 1 = -1
- Python interprets cwords[-1] as the LAST element (negative indexing)
- No IndexError is raised if cwords is non-empty
- This causes incorrect behavior

The bug is REAL but only occurs in the edge case where COMP_CWORD=0,
which would happen if bash completion is triggered at the command name position.
""")