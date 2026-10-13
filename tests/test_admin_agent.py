from langchain_core.language_models.fake_chat_models import FakeListChatModel

from parking_assistant.admin_agent import AdminAgent, OfflineAdminModel
from tests.conftest import make_request


def test_escalation_uses_tool_to_enqueue_request(service, clock):
    r = service.submit("s", make_request(clock))
    message = AdminAgent(service).escalate(r.id)
    assert service.get(r.id).queued_at is not None
    assert "waiting" in message


def test_agent_tools_cannot_decide(service, clock):
    model = OfflineAdminModel()
    agent = AdminAgent(service, model)
    r = service.submit("s", make_request(clock))
    agent.escalate(r.id)
    assert service.get(r.id).status.value == "pending"


def test_report_decision_reflects_human_rejection(service, clock):
    r = service.submit("s", make_request(clock))
    service.reject(r.id, "admin", "full")
    assert "rejected" in AdminAgent(service).report_decision(r.id)


def test_model_that_claims_false_success_is_overridden(service, clock):
    r = service.submit("s", make_request(clock))
    service.approve(r.id, "admin")  # approved but not yet recorded
    liar = FakeListChatModel(responses=["Your reservation was approved and recorded!"])

    class LiarWithTools(FakeListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    message = AdminAgent(
        service, LiarWithTools(responses=liar.responses)
    ).report_decision(r.id)
    assert "still in progress" in message


def test_model_that_skips_tool_still_results_in_queue_entry(service, clock):
    class Lazy(FakeListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    r = service.submit("s", make_request(clock))
    AdminAgent(service, Lazy(responses=["Done."])).escalate(r.id)
    assert service.get(r.id).queued_at is not None
