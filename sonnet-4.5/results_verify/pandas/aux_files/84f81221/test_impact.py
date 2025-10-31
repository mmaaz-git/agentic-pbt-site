#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from pandas.io.formats.printing import adjoin, _EastAsianTextAdjustment

# Reproduce the exact test case from test_adjoin_unicode
data = [["あ", "b", "c"], ["dd", "ええ", "ff"], ["ggg", "hhh", "いいい"]]

adj = _EastAsianTextAdjustment()

# Test the adjoin method using the adjustment class
result = adj.adjoin(2, *data)
print("Result from adj.adjoin (uses custom strlen):")
print(result)
print()

# The adjustment class's adjoin method calls the main adjoin with strlen=self.len
# Let's verify what's happening inside

def tracking_strlen(s):
    width = adj.len(s)
    print(f"strlen called for '{s}' (width={width})")
    return width

print("\nTracking which strings get measured:")
print("="*50)
result2 = adjoin(2, *data, strlen=tracking_strlen)
print("="*50)
print()
print("Strings that were NOT measured (from list3):")
print("- 'ggg' (width=3)")
print("- 'hhh' (width=3)")
print("- 'いいい' (width=6)")
print()
print("Even though the output appears correct, the last column widths")
print("were calculated using len() instead of the East Asian width function.")
print("This happens to work because the justfunc still uses the correct")
print("width-aware justify function, which compensates for the incorrect length.")