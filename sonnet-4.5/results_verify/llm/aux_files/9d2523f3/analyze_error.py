import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

# Look at what happens with int() conversion
test_values = [None, "10", 10, "not_number", True, False, 1.5]

for val in test_values:
    print(f"int({repr(val)}): ", end="")
    try:
        result = int(val)
        print(f"Success -> {result}")
    except TypeError as e:
        print(f"TypeError: {e}")
    except ValueError as e:
        print(f"ValueError: {e}")
