# tests/test_agent_inferencer.py
from routesmith.predictor.agent_inferencer import AGENT_ROLES, AgentInferencer


class TestAgentInferencer:
    def setup_method(self):
        self.inferencer = AgentInferencer()

    def test_research_prompt(self):
        messages = [
            {"role": "system", "content": "You are a research assistant. Analyze papers and summarize findings."},
            {"role": "user", "content": "What is the latest on transformer architectures?"},
        ]
        role, confidence = self.inferencer.infer(messages)
        assert role == "research"
        assert confidence > 0.3

    def test_coding_prompt(self):
        messages = [
            {"role": "system", "content": "You are a Python coding assistant. Help users debug and implement algorithms."},
        ]
        role, confidence = self.inferencer.infer(messages)
        assert role == "coding"
        assert confidence > 0.3

    def test_customer_service_prompt(self):
        messages = [
            {"role": "system", "content": "You are a customer support agent for Acme Corp. Resolve customer issues politely."},
        ]
        role, confidence = self.inferencer.infer(messages)
        assert role == "customer_service"
        assert confidence > 0.3

    def test_unknown_prompt_returns_none(self):
        messages = [
            {"role": "system", "content": "xyz123 qwerty foo bar baz"},
        ]
        role, confidence = self.inferencer.infer(messages)
        assert role is None
        assert confidence == 0.0

    def test_empty_messages_returns_none(self):
        role, confidence = self.inferencer.infer([])
        assert role is None
        assert confidence == 0.0

    def test_general_assistant_prompt(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]
        role, confidence = self.inferencer.infer(messages)
        assert role == "general"
        assert confidence > 0.0

    def test_caches_by_system_prompt_hash(self):
        messages = [
            {"role": "system", "content": "You help with code and debugging Python programs."},
        ]
        result1 = self.inferencer.infer(messages)
        result2 = self.inferencer.infer(messages)
        assert result1 == result2
        assert len(self.inferencer._cache) == 1

    def test_role_ordinal_known_roles(self):
        for role in AGENT_ROLES:
            assert AgentInferencer.role_ordinal(role) >= 1

    def test_role_ordinal_none_returns_zero(self):
        assert AgentInferencer.role_ordinal(None) == 0
