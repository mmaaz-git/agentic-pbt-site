import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages')

import datetime
from pydantic.deprecated.json import timedelta_isoformat

td = datetime.timedelta(seconds=-1)

print(f"Input: timedelta(seconds=-1)")
print(f"Python internal representation: {td}")
print(f"Actual duration (total_seconds): {td.total_seconds()}")

iso_output = timedelta_isoformat(td)
print(f"ISO format output: {iso_output}")

print("\nProblem:")
print(f"  The ISO format '{iso_output}' means:")
print(f"  'negative (1 day + 23 hours + 59 minutes + 59 seconds)'")
print(f"  = -(86400 + 82800 + 3540 + 59) = -172799 seconds")
print(f"  But the actual timedelta is -1 second!")
print(f"  The correct ISO format should be: -PT1S")