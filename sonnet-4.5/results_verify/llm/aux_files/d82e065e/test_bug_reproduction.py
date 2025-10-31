import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

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
        self.id = "test"
        self.object = "chat.completion.chunk"
        self.model = "gpt-4"
        self.created = 1234567890
        self.index = 0

# Test the bug as reported
chunks = [
    MockChunk([MockChoice(MockDelta(role="assistant", content="Hello"))]),
    MockChunk([MockChoice(MockDelta(role=None, content=" world"))]),
]

result = combine_chunks(chunks)
print(f"Content: {result['content']}")
print(f"Role: {result['role']}")
print(f"Expected role: assistant")
print(f"Bug confirmed: {result['role'] != 'assistant'}")