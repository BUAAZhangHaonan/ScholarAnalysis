"""Conservative budgeting with an explicit, non-fragmentary prefix boundary."""
from dataclasses import dataclass
from math import ceil

@dataclass(frozen=True)
class PromptBudget:
    model_context_tokens: int = 256000
    response_headroom_tokens: int = 16000
    tool_headroom_tokens: int = 4000
    message_overhead_tokens: int = 24

    @property
    def max_input_tokens(self):
        return max(1, self.model_context_tokens-self.response_headroom_tokens-self.tool_headroom_tokens)

    def estimate_text(self, text):
        return max(1, ceil(len(text.encode("utf-8"))/2)) if text else 0

    def estimate_messages(self, messages):
        return sum(self.message_overhead_tokens+self.estimate_text(str(m.get("role", "")))+
                   self.estimate_text(str(m.get("content", ""))) for m in messages)

    def truncate_text(self, text, max_tokens):
        if max_tokens <= 0:
            return ""
        if self.estimate_text(text) <= max_tokens:
            return text
        low, high = 0, len(text)
        while low < high:
            mid = (low+high+1)//2
            if self.estimate_text(text[:mid]) <= max_tokens:
                low = mid
            else:
                high = mid-1
        # Keep complete Markdown blocks: avoid slicing a formula/table paragraph.
        boundary = text.rfind("\n\n", 0, low+1)
        return text[:boundary] if boundary > 0 else ""
