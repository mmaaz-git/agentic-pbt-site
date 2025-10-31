import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

# Simulate the logic from lines 1067-1075
def test_py_prefix_logic():
    args = ['py:x=42', 'py:my_var=100', 'regular_var=test']
    vars = {}

    for value in args:
        if '=' not in value:
            print(f'Bad argument: {value!r}')
            continue
        name, value = value.split('=', 1)
        if name.startswith('py:'):
            # This is line 1073 - the bug
            name = name[:3]  # BUG: Should be name[3:]
            value = eval(value)
        vars[name] = value

    print("Variables dictionary after processing:")
    print(vars)
    print()

    # Check what we got
    if 'py:' in vars:
        print("BUG CONFIRMED: Variable name 'py:' exists in dictionary")
        print(f"  Value for 'py:': {vars['py:']}")

    if 'x' in vars:
        print("Variable 'x' exists in dictionary")
        print(f"  Value for 'x': {vars['x']}")
    else:
        print("Variable 'x' MISSING from dictionary")

    if 'my_var' in vars:
        print("Variable 'my_var' exists in dictionary")
        print(f"  Value for 'my_var': {vars['my_var']}")
    else:
        print("Variable 'my_var' MISSING from dictionary")

    print()
    print("Expected behavior:")
    print("  vars should contain: {'x': 42, 'my_var': 100, 'regular_var': 'test'}")
    print("Actual behavior:")
    print(f"  vars contains: {vars}")

test_py_prefix_logic()

print("\n" + "="*60)
print("Now testing with the FIX (name[3:] instead of name[:3]):")
print("="*60 + "\n")

def test_py_prefix_logic_fixed():
    args = ['py:x=42', 'py:my_var=100', 'regular_var=test']
    vars = {}

    for value in args:
        if '=' not in value:
            print(f'Bad argument: {value!r}')
            continue
        name, value = value.split('=', 1)
        if name.startswith('py:'):
            # FIXED: Using name[3:] to remove the prefix
            name = name[3:]  # FIXED: Remove first 3 chars
            value = eval(value)
        vars[name] = value

    print("Variables dictionary after processing (FIXED):")
    print(vars)
    print()

    # Check what we got
    if 'py:' in vars:
        print("Variable name 'py:' exists in dictionary")
        print(f"  Value for 'py:': {vars['py:']}")
    else:
        print("Variable 'py:' NOT in dictionary (correct)")

    if 'x' in vars:
        print("Variable 'x' exists in dictionary (correct)")
        print(f"  Value for 'x': {vars['x']}")

    if 'my_var' in vars:
        print("Variable 'my_var' exists in dictionary (correct)")
        print(f"  Value for 'my_var': {vars['my_var']}")

    print()
    print("Expected behavior:")
    print("  vars should contain: {'x': 42, 'my_var': 100, 'regular_var': 'test'}")
    print("Actual behavior:")
    print(f"  vars contains: {vars}")

test_py_prefix_logic_fixed()