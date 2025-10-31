from hypothesis import given, strategies as st
from llm.default_plugins.openai_models import combine_chunks
from unittest.mock import Mock


def make_chunk(role, content):
    chunk = Mock()
    chunk.usage = None
    chunk.id = None
    chunk.object = None
    chunk.model = None
    chunk.created = None
    chunk.index = None

    choice = Mock()
    delta = Mock()
    delta.role = role
    delta.content = content
    choice.delta = delta
    choice.finish_reason = None
    choice.logprobs = None
    chunk.choices = [choice]
    return chunk


@given(st.lists(st.text(), min_size=2, max_size=10))
def test_combine_chunks_preserves_role(contents):
    chunks = [make_chunk("assistant" if i == 0 else None, content)
              for i, content in enumerate(contents)]
    result = combine_chunks(chunks)
    assert result["role"] == "assistant"


if __name__ == "__main__":
    test_combine_chunks_preserves_role()