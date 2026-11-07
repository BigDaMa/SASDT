import csv
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import tiktoken

from gdt.config import Config, ExperimentConfig
from gdt.experiments import write_summary
from gdt.export_results import load_results
from gdt.models import resolve_openai_model
from gdt.pipeline import PipelineRunner
from gdt.pricing import (
    CostTracker,
    RequestTokens,
    cost_session,
    count_tokens,
    track_request,
    write_pricing,
)
from gdt.results import EvaluationResult, RunResult, append_jsonl
from LLMUtil.ollama import OLLAMAUtil
from LLMUtil.providers import OpenAIUtil, request_split_with_retry
from LLMUtil.semantic import OpenAIUtil_Prompt
from Utils.BenchmarkCache import BenchmarkCache


@pytest.mark.parametrize("alias,canonical", [
    ("GPT-5.1", "gpt-5.1"), ("GPT5.1", "gpt-5.1"), ("gpt 5.1", "gpt-5.1"),
    ("GpT_5.4", "gpt-5.4"), ("GPT5.4", "gpt-5.4"), ("gpt 5.4", "gpt-5.4"),
    ("GPT6", "gpt-6-astra"), ("gpt 6", "gpt-6-astra"), ("GPT-6", "gpt-6-astra"),
    ("GPT_6_ASTRA", "gpt-6-astra"),
])
def test_aliases_reach_config_and_api(alias, canonical):
    config = ExperimentConfig(model=alias, llm_model=alias)
    assert config.model == config.llm_model == canonical
    client = Mock()
    client.responses.create.return_value.output_text = "answer"
    with patch.object(OpenAIUtil, "get_client", return_value=client):
        OpenAIUtil.create_response("question", instructions="instructions", model=alias)
    assert client.responses.create.call_args.kwargs["model"] == canonical


def test_aliases_do_not_rewrite_other_models_or_snapshots():
    for model in ("GPT5.2", "gpt-5.1-2025-11-13", "qwen3.5-4b", "gpt-4.1", "GPT5.4-mini"):
        assert resolve_openai_model(model) == model


def test_tiktoken_counts_unicode_whitespace_and_special_text():
    text = " Alice\n東京 👋 <|endoftext|>  "
    for model in ("GPT5.4", "qwen3.5:4b"):
        expected = len(tiktoken.get_encoding("o200k_base").encode(text, disallowed_special=()))
        assert count_tokens(text, model) == expected


def test_context_threshold_is_per_request_and_applies_to_both_rates():
    costs = CostTracker()
    costs.record(RequestTokens("GPT5.4", 272_999, 10))
    costs.record(RequestTokens("GPT5.4", 273_000, 10))
    costs.record(RequestTokens("GPT5.4", 1, 400_000))
    assert costs.costs["llm"] == Decimal("272999") * Decimal("2.5") + 150 + (
        273_000 * 5 + 225 + Decimal("2.5") + 400_000 * 15
    )
    # A large local-model prompt still uses its supplied short-context rate.
    costs.record(RequestTokens("qwen3.5:4b", 273_000, 10))
    assert costs.costs["slm"] == Decimal("21841.2")


def test_all_supplied_prices_and_family_totals():
    costs = CostTracker()
    for model in ("GPT5.1", "GPT5.4", "GPT6", "qwen3.5:4b", "granite4.2-3b", "ministral3-3b"):
        costs.record(RequestTokens(model, 100, 10))
    assert costs.result_fields() == {
        "llm_input_tokens": 300, "llm_output_tokens": 30,
        "slm_input_tokens": 300, "slm_output_tokens": 30,
        "cf_llm": 2125.0, "cf_slm": 24.4, "cf_total": 2149.4,
    }


def test_openai_retries_count_actual_instructions_and_raw_output():
    client = Mock()
    client.with_options.return_value = client
    client.responses.create.side_effect = [
        RuntimeError("offline"), SimpleNamespace(output_text="  answer\n"),
    ]
    costs = CostTracker()
    with cost_session(costs), patch.object(OpenAIUtil, "get_client", return_value=client), \
         patch("LLMUtil.providers.time.sleep"):
        assert OpenAIUtil.create_response(
            "question", instructions="instructions", model="GPT5.4", num_retry=2,
        ) == "answer"
    request_tokens = count_tokens("question", "gpt-5.4") + count_tokens("instructions", "gpt-5.4")
    assert costs.tokens["llm_input_tokens"] == 2 * request_tokens
    assert costs.tokens["llm_output_tokens"] == count_tokens("  answer\n", "gpt-5.4")
    client.with_options.assert_called_once_with(max_retries=0)


def test_rejected_responses_and_retries_both_count():
    client = Mock()
    client.with_options.return_value = client
    client.responses.create.side_effect = [
        SimpleNamespace(output_text="bad"), SimpleNamespace(output_text="a[ITEM]b"),
    ]
    costs = CostTracker()
    with cost_session(costs), patch.object(OpenAIUtil, "get_client", return_value=client):
        request_split_with_retry(
            lambda: OpenAIUtil.create_response("question", instructions="split", model="GPT5.1"),
            prompt="question", expected_count=2, operation="test", num_retry=1,
        )
    assert costs.tokens["llm_input_tokens"] == 2 * (
        count_tokens("question", "GPT5.1") + count_tokens("split", "GPT5.1")
    )
    assert costs.tokens["llm_output_tokens"] == (
        count_tokens("bad", "GPT5.1") + count_tokens("a[ITEM]b", "GPT5.1")
    )


@pytest.mark.parametrize("mode", ["gen", "chat"])
def test_ollama_counts_final_text_and_excludes_lifecycle_and_server_counts(mode):
    reply = Mock()
    reply.json.return_value = {
        "response": " answer ", "thinking": "ignored " * 100,
        "message": {"content": " answer ", "thinking": "ignored " * 100},
        "prompt_eval_count": 99999, "eval_count": 99999,
    }
    costs = CostTracker()
    with cost_session(costs), patch.object(OLLAMAUtil, "__MODEL__", "qwen3.5-4b"), \
         patch.object(OLLAMAUtil, "__MODE__", mode), patch("requests.post", return_value=reply):
        OLLAMAUtil.load_model()
        assert OLLAMAUtil.get_response_batch(["prompt"]) == ["answer"]
        OLLAMAUtil.unload_model()
    assert costs.tokens["slm_input_tokens"] == count_tokens("prompt", "qwen3.5:4b")
    assert costs.tokens["slm_output_tokens"] == count_tokens(" answer ", "qwen3.5:4b")
    assert costs.result_fields()["cf_llm"] == 0


def test_cached_response_adds_no_cost():
    BenchmarkCache.clear()
    client = Mock()
    client.with_options.return_value = client
    client.responses.create.return_value.output_text = "answer"
    costs = CostTracker()
    with cost_session(costs), patch.object(Config, "money_saver", True), \
         patch.object(OpenAIUtil, "get_client", return_value=client):
        for model in ("GPT5.4", "gpt-5.4"):
            OpenAIUtil_Prompt._cached_response("test", "question", "instructions", model=model)
    assert client.responses.create.call_count == 1
    assert costs.tokens["llm_input_tokens"] == 2
    BenchmarkCache.clear()


def test_pipeline_costs_survive_failure_and_reset_between_datasets(tmp_path):
    def run(_path):
        with track_request("GPT5.4", "question") as usage:
            usage.record_output("answer")
        raise RuntimeError("downstream failure")

    with patch.object(PipelineRunner, "_run", side_effect=run):
        runner = PipelineRunner(ExperimentConfig(output_directory=tmp_path))
        first, second = runner.run("first.csv"), runner.run("second.csv")
    assert not first.succeeded and not second.succeeded
    assert first.cf_llm == second.cf_llm == 17.5
    assert first.cf_slm == second.cf_slm == 0


def test_four_column_report_sums_datasets_modes_repetitions_and_failures(tmp_path):
    def result(dataset, model="GPT5.4", count=1, **kwargs):
        return RunResult(dataset, model, "gen", count, 1, llm_model="GPT5.1", **kwargs)

    results = [
        result("a.csv", cf_llm=10, cf_slm=0),
        result("b.csv", model="gpt-5.4", cf_llm=20, cf_slm=0, error="failure"),
        result("a.csv", repetition=2, cf_llm=10, cf_slm=0),
        result("a.csv", model="qwen3.5-4b", cf_llm=10, cf_slm=1.2),
        result("a.csv", count=2, cf_llm=25, cf_slm=0),
    ]
    write_pricing(results, tmp_path)
    with (tmp_path / "pricing.csv").open(newline="") as handle:
        assert list(csv.reader(handle)) == [
            ["Experiment", "CF-LLM", "CF-SLM", "CF-Total"],
            ["GPT5.1_GPT5.4_1", "40", "0", "40"],
            ["GPT5.1_qwen3.5-4b_1", "10", "1.2", "11.2"],
            ["GPT5.1_GPT5.4_2", "25", "0", "25"],
        ]


def test_costs_round_trip_csv_and_jsonl_and_old_results_stay_unmeasured(tmp_path):
    result = RunResult("a.csv", "GPT5.4", "gen", 1, 1,
                       evaluation=EvaluationResult(1, 1, 0),
                       cf_llm=17.5, cf_slm=1.2, cf_total=18.7,
                       llm_input_tokens=1, slm_output_tokens=10)
    write_summary([result], tmp_path)
    append_jsonl(tmp_path / "runs.jsonl", result)
    for path in (tmp_path / "summary.csv", tmp_path / "runs.jsonl"):
        loaded = load_results(path)[0]
        assert loaded == result
    (tmp_path / "old.csv").write_text(
        "dataset,model,mode,num_examples,duration_seconds,error\na.csv,gpt-5.1,gen,1,1,failed\n",
    )
    write_pricing(load_results(tmp_path / "old.csv"), tmp_path)
    assert "N/A,N/A,N/A" in (tmp_path / "pricing.csv").read_text()


def test_unpriced_model_is_not_reported_as_free():
    costs = CostTracker()
    with pytest.warns(UserWarning, match="No pricing"):
        costs.record(RequestTokens("unpriced-local-model", 10, 1))
    assert costs.result_fields()["cf_slm"] is None
    assert costs.result_fields()["cf_llm"] == 0
    assert costs.result_fields()["cf_total"] is None
