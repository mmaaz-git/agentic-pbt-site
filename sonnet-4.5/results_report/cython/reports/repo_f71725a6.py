import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

# Simulating the exact bug from the fill_command function
# This is the problematic code from line 1072-1075 of Cython/Tempita/_tempita.py

def test_py_prefix_bug():
    """Demonstrate the bug in Cython.Tempita.fill_command py: prefix handling"""

    # Test case 1: Single py: prefixed variable
    print("Test Case 1: Single py: prefixed variable")
    print("-" * 40)

    vars = {}
    arg_string = "py:my_var=42"

    # Simulate parsing the argument (lines 1071-1075)
    name, value = arg_string.split('=', 1)
    print(f"Original argument: {arg_string}")
    print(f"After split: name='{name}', value='{value}'")

    if name.startswith('py:'):
        name = name[:3]  # BUG: This keeps 'py:' instead of removing it
        value = eval(value)

    vars[name] = value

    print(f"Variable name stored in dict: '{name}'")
    print(f"Variable value stored: {value}")
    print(f"Result: vars = {vars}")
    print(f"Expected: vars = {{'my_var': 42}}")
    print(f"Actual:   vars = {vars}")
    print()

    # Test case 2: Multiple py: prefixed variables showing overwrite
    print("Test Case 2: Multiple py: prefixed variables")
    print("-" * 40)

    vars = {}
    args = ["py:x=10", "py:y=20", "py:z=30"]

    print(f"Arguments to parse: {args}")

    for arg_string in args:
        name, value = arg_string.split('=', 1)
        if name.startswith('py:'):
            name = name[:3]  # BUG: All become 'py:'
            value = eval(value)
        vars[name] = value
        print(f"  After parsing '{arg_string}': vars = {vars}")

    print()
    print(f"Expected: vars = {{'x': 10, 'y': 20, 'z': 30}}")
    print(f"Actual:   vars = {vars}")
    print(f"Bug: All variables overwrite each other under key 'py:'!")
    print()

    # Test case 3: Compare with correct implementation
    print("Test Case 3: Correct implementation (what it should be)")
    print("-" * 40)

    vars = {}
    args = ["py:x=10", "py:y=20", "py:z=30"]

    print(f"Arguments to parse: {args}")

    for arg_string in args:
        name, value = arg_string.split('=', 1)
        if name.startswith('py:'):
            name = name[3:]  # FIX: Remove 'py:' prefix correctly
            value = eval(value)
        vars[name] = value
        print(f"  After parsing '{arg_string}': vars = {vars}")

    print()
    print(f"Result with fix: vars = {vars}")
    print(f"This is what the code should produce!")

if __name__ == "__main__":
    test_py_prefix_bug()