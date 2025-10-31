#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from pydantic import BaseModel
from typing import Optional
from hypothesis import given, strategies as st

class MinimalOptions(BaseModel):
    value1: Optional[int] = None
    value2: Optional[str] = None

def not_nulls(data):
    return {key: value for key, value in data if value is not None}

@given(
    st.builds(
        MinimalOptions,
        value1=st.one_of(st.none(), st.integers()),
        value2=st.one_of(st.none(), st.text())
    )
)
def test_not_nulls_fails_with_pydantic_models(options):
    try:
        result = not_nulls(options)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "not enough values to unpack" in str(e)
        print(f"Test passed - ValueError raised as expected: {e}")

if __name__ == "__main__":
    # Run the hypothesis test
    print("Running hypothesis test...")
    test_not_nulls_fails_with_pydantic_models()
    print("\nHypothesis test completed successfully - all generated inputs caused ValueError")