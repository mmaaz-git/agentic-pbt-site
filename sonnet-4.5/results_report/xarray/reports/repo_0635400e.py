import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import datetime
import pandas as pd
from xarray.compat.pdcompat import default_precision_timestamp

# Create a datetime beyond nanosecond precision range (after 2262-04-11)
dt = datetime.datetime(2263, 1, 1, 0, 0)

# Show that pd.Timestamp can handle this date
ts = pd.Timestamp(dt)
print(f"pd.Timestamp works: {ts}, unit={ts.unit}")

# Now try with default_precision_timestamp - this will crash
try:
    result = default_precision_timestamp(dt)
    print(f"default_precision_timestamp result: {result}, unit={result.unit}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")