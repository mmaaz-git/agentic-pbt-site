import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from llm.default_plugins.openai_models import combine_chunks

class MockDelta:
    def __init__(self, role=None, content=None):
        self.role = role
        self.content = content

class MockChoice:
    def __init__(self, delta, finish_reason=None):
        self.delta = delta
        self.finish_reason = finish_reason
        self.logprobs = None

class MockChunk:
    def __init__(self, choices):
        self.choices = choices
        self.usage = None

@given(st.lists(st.text(), min_size=1))
def test_combine_chunks_preserves_role(content_pieces):
    """Property: role from first chunk should be preserved"""
    chunks = []
    for i, content in enumerate(content_pieces):
        role = "assistant" if i == 0 else None
        chunks.append(MockChunk([MockChoice(MockDelta(role=role, content=content))]))

    result = combine_chunks(chunks)
    assert result['role'] == "assistant", f"Expected 'assistant', got {result['role']}"

if __name__ == "__main__":
    test_combine_chunks_preserves_role()