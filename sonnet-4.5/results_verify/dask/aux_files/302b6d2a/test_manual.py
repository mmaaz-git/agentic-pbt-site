from dask.diagnostics import ProgressBar
from dask.threaded import get
from operator import add
import io
import time

output = io.StringIO()
dsk = {'x': 1, 'y': (add, 'x', 10)}

pbar = ProgressBar(dt=-0.1, out=output)

with pbar:
    result = get(dsk, 'y')
    time.sleep(0.2)

print(f"Result: {result}")