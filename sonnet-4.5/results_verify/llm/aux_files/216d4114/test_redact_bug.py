import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from llm.default_plugins.openai_models import redact_data
import copy


@given(st.text(min_size=1))
def test_redact_data_should_not_mutate_input_image_url(data_content):
    original = {"image_url": {"url": f"data:image/png;base64,{data_content}"}}
    original_copy = copy.deepcopy(original)

    result = redact_data(original)

    assert original == original_copy, (
        f"redact_data mutated its input! "
        f"Before: {original_copy}, After: {original}"
    )

if __name__ == "__main__":
    test_redact_data_should_not_mutate_input_image_url()