from planning_eval.person2_benchmark import (
    InstrumentedLLM,
    UngroundedActionEnvironment,
    summarize_person2_records,
)


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeRunner:
    def invoke(self, _messages, **_kwargs):
        return {"score": 0.8}


class FakeLLM:
    def invoke(self, _messages, **_kwargs):
        return FakeResponse("restock product 4")

    def with_structured_output(self, *_args, **_kwargs):
        return FakeRunner()


def test_instrumented_llm_records_normal_and_structured_calls():
    llm = InstrumentedLLM(FakeLLM())

    llm.invoke([("human", "choose a restock action")])
    llm.with_structured_output(dict, method="json_mode").invoke(
        [("human", "score the action")]
    )

    assert llm.metrics.calls == 2
    assert llm.metrics.input_tokens > 0
    assert llm.metrics.output_tokens > 0
    assert llm.metrics.latency_ms >= 0


def test_ungrounded_baseline_can_accept_an_unsupported_restock_action():
    feedback = UngroundedActionEnvironment("restock").evaluate(
        '{"action":"restock","product_id":999,"quantity_change":20,"user_id":3}'
    )

    assert feedback.success is True


def test_summary_aggregates_measured_records():
    summary = summarize_person2_records(
        [
            {
                "method": "reflexion",
                "grounded": True,
                "success": True,
                "metrics": {"calls": 2, "total_tokens": 30, "latency_ms": 12.0},
            },
            {
                "method": "reflexion",
                "grounded": True,
                "success": False,
                "metrics": {"calls": 4, "total_tokens": 50, "latency_ms": 20.0},
            },
        ]
    )

    assert summary == [
        {
            "method": "reflexion",
            "grounded": True,
            "cases": 2,
            "success_rate": 0.5,
            "avg_calls": 3.0,
            "avg_total_tokens": 40.0,
            "avg_latency_ms": 16.0,
        }
    ]