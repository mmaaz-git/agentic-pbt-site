import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Build.Inline import cymeit

print("Testing cymeit with a zero float timer...")

counter = [0]
max_calls = 10000

def zero_float_timer():
    counter[0] += 1
    if counter[0] > max_calls:
        raise RuntimeError(f"INFINITE LOOP: autorange() called timer {max_calls}+ times")
    return 0.0

code = "x = 1"

try:
    print(f"Running cymeit with timer that always returns 0.0...")
    result = cymeit(code, timer=zero_float_timer, repeat=3)
    print(f"Test completed successfully. Timer was called {counter[0]} times")
    print(f"Result: {result}")
except RuntimeError as e:
    if "INFINITE LOOP" in str(e):
        print(f"BUG CONFIRMED: {e}")
        print(f"The autorange() function has no termination check for float timers returning < 0.2")
    else:
        print(f"Other error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")