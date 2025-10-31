import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from pydantic import BaseModel
from pydantic.experimental.pipeline import transform


@given(st.integers(min_value=10, max_value=100))
def test_constraint_application_consistency(x):
    class ModelGe(BaseModel):
        value: int = transform(lambda v: v).ge(5)

    class ModelGt(BaseModel):
        value: int = transform(lambda v: v).gt(4)

    m_ge = ModelGe(value=x)
    m_gt = ModelGt(value=x)

    assert m_ge.value == x
    assert m_gt.value == x

if __name__ == "__main__":
    test_constraint_application_consistency()