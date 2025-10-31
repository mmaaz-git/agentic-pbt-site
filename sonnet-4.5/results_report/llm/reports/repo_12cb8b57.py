from llm.default_plugins.openai_models import combine_chunks
from unittest.mock import Mock


def make_chunk(role, content):
    chunk = Mock()
    chunk.usage = None
    chunk.id = chunk.object = chunk.model = chunk.created = chunk.index = None
    choice = Mock()
    delta = Mock()
    delta.role = role
    delta.content = content
    choice.delta = delta
    choice.finish_reason = None
    choice.logprobs = None
    chunk.choices = [choice]
    return chunk


# Test case with two chunks - first has role, second doesn't
chunks = [
    make_chunk("assistant", "Hello"),
    make_chunk(None, " World"),
]

result = combine_chunks(chunks)
print(f"Result: {result}")
print(f"Role: {result['role']}")
print(f"Content: {result['content']}")

# This assertion should pass, but it fails due to the bug
try:
    assert result["role"] == "assistant", f"Expected role='assistant', got role={result['role']}"
    print("PASS: Role preserved correctly")
except AssertionError as e:
    print(f"FAIL: {e}")