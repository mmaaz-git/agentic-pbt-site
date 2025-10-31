#!/usr/bin/env python3
"""Test if COMP_CWORD=0 can occur in practice with Django's autocomplete."""

print("=" * 60)
print("Testing if COMP_CWORD=0 can occur in Django's autocomplete")
print("=" * 60)

print("""
Analyzing when COMP_CWORD could be 0:

1. According to bash documentation and our tests:
   - COMP_CWORD is the index into COMP_WORDS array
   - COMP_CWORD=0 means cursor is at the command name itself

2. Django's autocomplete() is called when DJANGO_AUTO_COMPLETE is set
   - This is set by Django's bash completion script
   - The script is sourced to enable completion for django-admin/manage.py

3. Looking at line 303 in Django:
   cwords = os.environ["COMP_WORDS"].split()[1:]

   This removes the command name from the words array.
   So if COMP_WORDS = "django-admin", then cwords = []

4. The problematic case would be:
   - COMP_WORDS = "django-admin somearg"
   - COMP_CWORD = 0 (cursor at 'django-admin' position)
   - After split()[1:], cwords = ['somearg']
   - Then cwords[0-1] = cwords[-1] = 'somearg' (WRONG!)

HOWEVER: Can bash actually set COMP_CWORD=0 when calling a completion function?
""")

print("\nChecking bash completion behavior...")
print("-" * 40)

print("""
From bash manual and standard completion behavior:
- When you type 'command <TAB>', COMP_CWORD is typically 1 or higher
- COMP_CWORD=0 would mean completing the command name itself
- But Django's completion is registered FOR a specific command (django-admin)
- So the completion function is only called AFTER the command name

CONCLUSION: In normal bash completion usage:
- Django's autocomplete() is only invoked AFTER 'django-admin' is typed
- At that point, COMP_CWORD should be >= 1
- COMP_CWORD=0 entering Django's function seems unlikely in practice

But what if:
1. User manually sets COMP_CWORD=0 in environment (malicious/testing)
2. Some edge case in bash completion internals
3. Bug in the bash completion script that calls Django
""")

# Let's check what would actually happen
print("\n" + "=" * 60)
print("TESTING THE ACTUAL BUG SCENARIO")
print("=" * 60)

test_scenarios = [
    {
        "COMP_WORDS": "django-admin",
        "COMP_CWORD": 0,
        "description": "Edge case: COMP_CWORD=0 with just command"
    },
    {
        "COMP_WORDS": "django-admin migrate",
        "COMP_CWORD": 0,
        "description": "Bug case: COMP_CWORD=0 with arguments"
    },
    {
        "COMP_WORDS": "django-admin migrate --database=default",
        "COMP_CWORD": 0,
        "description": "Bug case: COMP_CWORD=0 with multiple arguments"
    }
]

for scenario in test_scenarios:
    print(f"\n{scenario['description']}")
    print("-" * 40)

    comp_words = scenario["COMP_WORDS"]
    comp_cword = scenario["COMP_CWORD"]

    # Django's logic
    cwords = comp_words.split()[1:]

    print(f"COMP_WORDS: '{comp_words}'")
    print(f"COMP_CWORD: {comp_cword}")
    print(f"cwords after split()[1:]: {cwords}")

    try:
        curr = cwords[comp_cword - 1]
        print(f"curr = cwords[{comp_cword - 1}] = '{curr}'")
        if comp_cword == 0 and len(cwords) > 0:
            print(f"⚠️  BUG CONFIRMED: Got '{curr}' instead of ''")
    except IndexError:
        curr = ""
        print(f"curr = '' (IndexError caught)")

print("\n" + "=" * 60)
print("FINAL ASSESSMENT")
print("=" * 60)
print("""
The bug is TECHNICALLY CORRECT:
- When COMP_CWORD=0 and there are arguments after the command name
- Python's negative indexing causes cwords[-1] to return the last element
- This violates the intended behavior of returning empty string

However, COMP_CWORD=0 should NOT occur in normal Django usage because:
- Bash completion for django-admin is only triggered AFTER typing the command
- At that point COMP_CWORD would be 1 or higher

This is likely:
1. A defensive programming issue (edge case not properly handled)
2. Could be exploited if someone manually sets COMP_CWORD=0
3. Might occur in unusual bash configurations or completion scripts
""")