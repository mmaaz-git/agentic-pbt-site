import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

import pandas as pd

text = '0'
start = 1
stop = 0
repl = ''

s = pd.Series([text])
pandas_result = s.str.slice_replace(start, stop, repl)[0]
python_result = text[:start] + repl + text[stop:]

print(f"Text: '{text}'")
print(f"start={start}, stop={stop}, repl='{repl}'")
print(f"Python: '{python_result}'")
print(f"Pandas: '{pandas_result}'")
print(f"Match: {pandas_result == python_result}")

# Let's also check what Python slicing does in this case
print("\nDebug info:")
print(f"text[:start] = '{text[:start]}' (text[:1])")
print(f"text[stop:] = '{text[stop:]}' (text[0:])")
print(f"text[start:stop] = '{text[start:stop]}' (text[1:0]) - this is empty when start > stop")