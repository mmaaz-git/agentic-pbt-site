import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Build.Inline import cymeit

counter = [0]

def zero_float_timer():
    counter[0] += 1
    if counter[0] > 10000:
        raise RuntimeError("INFINITE LOOP: autorange() called timer 10000+ times")
    return 0.0

code = "x = 1"

try:
    cymeit(code, timer=zero_float_timer, repeat=3)
except RuntimeError as e:
    if "INFINITE LOOP" in str(e):
        print(f"BUG CONFIRMED: {e}")
        print(f"autorange() has no termination check for float timers")
        print(f"Timer was called {counter[0]} times before safety limit")