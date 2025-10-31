import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

import pandas as pd

# Test various cases with Python slicing and pandas
test_cases = [
    ('0', 1, 0, ''),
    ('abc', 2, 1, 'X'),
    ('hello', 3, 2, 'Y'),
    ('test', 4, 0, 'Z'),
    ('12345', 5, 2, ''),
]

for text, start, stop, repl in test_cases:
    # Python behavior
    python_result = text[:start] + repl + text[stop:]

    # Pandas behavior
    s = pd.Series([text])
    pandas_result = s.str.slice_replace(start, stop, repl)[0]

    match = "✓" if pandas_result == python_result else "✗"
    print(f"{match} text='{text}', start={start}, stop={stop}, repl='{repl}'")
    print(f"  Python: '{python_result}'")
    print(f"  Pandas: '{pandas_result}'")
    print(f"  text[{start}:{stop}] = '{text[start:stop]}'")
    print()