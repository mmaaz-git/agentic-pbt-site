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
assert pandas_result == python_result