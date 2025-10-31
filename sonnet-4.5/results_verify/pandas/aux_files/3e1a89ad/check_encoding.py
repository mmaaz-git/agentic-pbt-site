#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from pandas._config import get_option

print(f"Default display.encoding: {get_option('display.encoding')}")