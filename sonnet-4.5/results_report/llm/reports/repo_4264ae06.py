from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Delta:
    role: Optional[str]
    content: Optional[str]

@dataclass
class Choice:
    delta: Delta
    finish_reason: Optional[str]

    @property
    def logprobs(self):
        return None

@dataclass
class Chunk:
    choices: List[Choice]
    usage: Optional[object] = None

def combine_chunks(chunks: List) -> dict:
    content = ""
    role = None
    finish_reason = None
    # If any of them have log probability, we're going to persist
    # those later on
    logprobs = []
    usage = {}

    for item in chunks:
        if item.usage:
            usage = item.usage.model_dump()
        for choice in item.choices:
            if choice.logprobs and hasattr(choice.logprobs, "top_logprobs"):
                logprobs.append(
                    {
                        "text": choice.text if hasattr(choice, "text") else None,
                        "top_logprobs": choice.logprobs.top_logprobs,
                    }
                )

            if not hasattr(choice, "delta"):
                content += choice.text
                continue
            role = choice.delta.role
            if choice.delta.content is not None:
                content += choice.delta.content
            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason

    # Imitations of the OpenAI API may be missing some of these fields
    combined = {
        "content": content,
        "role": role,
        "finish_reason": finish_reason,
        "usage": usage,
    }
    if logprobs:
        combined["logprobs"] = logprobs
    if chunks:
        for key in ("id", "object", "model", "created", "index"):
            value = getattr(chunks[0], key, None)
            if value is not None:
                combined[key] = value

    return combined

# Test case mimicking OpenAI streaming behavior
chunks = [
    # First chunk typically contains the role
    Chunk(choices=[Choice(delta=Delta(role="assistant", content="Hello"), finish_reason=None)]),
    # Subsequent chunks have role=None but contain content
    Chunk(choices=[Choice(delta=Delta(role=None, content=" world"), finish_reason=None)]),
]

result = combine_chunks(chunks)
print(f"Result role: {result['role']}")
print(f"Result content: {result['content']}")
print(f"Expected role: 'assistant'")
print(f"Bug: role is {result['role']} instead of 'assistant'")