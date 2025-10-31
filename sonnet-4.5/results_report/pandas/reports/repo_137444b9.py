import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas.io.formats.css as css
import warnings

resolver = css.CSSResolver()

test_cases = ["1e-10pt", "1.5e-20px", "3e-100em"]

for test_case in test_cases:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = resolver.size_to_pt(test_case)
        print(f"Input: {test_case}, Result: {result}, Warnings: {len(w) > 0}")
        if len(w) > 0:
            print(f"  Warning message: {w[0].message}")