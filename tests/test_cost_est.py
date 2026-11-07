import csv
import random
import re
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import cost_est
from gdt.config import Config, ExperimentConfig
from gdt.data import PairDataset
from gdt.pricing import CostTracker, cost_session
from LLMUtil.ollama import OLLAMAUtil
from LLMUtil.providers import OpenAIUtil
from LLMUtil.ResponseParser import RParser
from LLMUtil.semantic import OpenAIUtil_Prompt


@pytest.fixture
def corpus(tmp_path):
    data, truth = tmp_path / "data", tmp_path / "truth"
    data.mkdir()
    truth.mkdir()
    rows = [
        ("Alice Smith", "Smith, Alice"), ("Bob Brown", "Brown, Bob"),
        ("Inactive Two", "Two, Inactive"), ("Inactive Three", "Three, Inactive"),
        ("Inactive Four", "Four, Inactive"), ("Carol Jones", "Jones, Carol"),
        ("DAN|WHITE", "WHITE, DAN"), ("EVE|GREEN", "GREEN, EVE"),
    ]
    path = data / "Names_0.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    groups = [[source for source, _ in rows if "|" not in source],
              [source for source, _ in rows if "|" in source],
              [target for _, target in rows]]
    (truth / "Names_cluster.txt").write_text(
        "\n[ITEM]\n".join("\n".join(group) for group in groups), encoding="utf-8",
    )
    blocks = []
    for source, target in rows:
        first, last = re.split(r"[ |]", source)
        # Deliberately unordered mappings; composition order comes from substrings.
        for original in (source, target):
            blocks.append(f"{original}\nlast_name:{last}\nfirst_name:{first}")
    (truth / "Names_split.txt").write_text("\n[ITEM]\n".join(blocks), encoding="utf-8")
    config = ExperimentConfig(data_directory=data, perfect_directory=truth,
                              output_directory=tmp_path / "results", num_examples=2,
                              llm_model="GPT5.4", model="GPT5.4")
    return PairDataset.from_csv(path), config


@pytest.fixture(autouse=True)
def forbid_network_and_experiments(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("cost estimation must not run experiments or contact a model")

    monkeypatch.setattr(OpenAIUtil, "get_client", forbidden)
    monkeypatch.setattr("requests.post", forbidden)
    monkeypatch.setattr(cost_est.PipelineRunner, "run", forbidden)
    yield
    Config.apply(ExperimentConfig())


def requests_for(dataset, config):
    truth = cost_est.EstimationTruth(dataset, config.perfect_directory, config.num_examples)
    return list(cost_est.plan_requests(dataset, config, truth))


def test_selected_strings_and_composition_order(corpus):
    dataset, config = corpus
    requests = requests_for(dataset, config)
    assert [r.phase for r in requests] == [
        "example compositions", "source compositions", "exact extraction",
    ]
    assert requests[0].query_ids == (0, 8, 1, 9)
    assert requests[1].query_ids == (5, 6, 7)
    assert requests[2].query_ids == (0, 1, 6, 8, 9)
    assert "Alice Smith\n[first_name][last_name]" in requests[0].response
    assert "Smith, Alice\n[last_name][first_name]" in requests[0].response
    assert "Alice Smith\nORIGINAL_STRING: Smith, Alice\n[ITEM]" in requests[0].input_texts[1]
    assert "last_name:Smith\nfirst_name:Alice" in requests[2].response
    for request in requests:
        assert "Inactive" not in "\n".join(request.input_texts) + request.response
        assert "Jones, Carol" not in "\n".join(request.input_texts) + request.response


def test_missing_truth_empties_only_matching_batch_body(corpus, capsys):
    dataset, config = corpus
    path = config.perfect_directory / "Names_split.txt"
    blocks = path.read_text().split("\n[ITEM]\n")
    path.write_text("\n[ITEM]\n".join(block for block in blocks
                                    if block.splitlines()[0] not in {"Brown, Bob", "DAN|WHITE"}))
    requests = requests_for(dataset, config)
    assert requests[0].response.endswith("ORIGINAL_STRING: Brown, Bob\n")
    assert "[first_name][last_name]" in requests[0].response
    assert "ORIGINAL_STRING: DAN|WHITE\n\n[ITEM]\n" in requests[1].response
    assert "ORIGINAL_STRING: EVE|GREEN\n[first_name][last_name]" in requests[1].response
    assert "ORIGINAL_STRING: DAN|WHITE\n\n[ITEM]\n" in requests[2].response
    assert "last_name:Smith\nfirst_name:Alice" in requests[2].response
    cost_est.estimate_dataset(dataset, config)
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("model,batch,local_split", [
    ("gpt-5.4", False, False), ("gpt-5.1", True, False),
    ("qwen3.5-4b", False, False), ("qwen3.5-4b", True, False),
    ("qwen3.5-4b", False, True), ("qwen3.5-4b", True, True),
])
def test_prompts_and_costs_match_real_lm_stages(corpus, monkeypatch, model, batch, local_split):
    dataset, config = corpus
    config = replace(config, model=model, language_model_batch=batch, slm_split=local_split)
    planned = requests_for(dataset, config)
    pending = iter(planned)
    observed = []

    def gpt_reply(**kwargs):
        expected = next(pending)
        observed.append((kwargs["model"], (kwargs["instructions"], kwargs["input"])))
        return SimpleNamespace(output_text=expected.response)

    def slm_reply(url, *, json, **kwargs):
        expected = next(pending)
        observed.append((json["model"], (json["prompt"],)))
        return SimpleNamespace(raise_for_status=lambda: None,
                               json=lambda: {"response": expected.response})

    client = Mock()
    client.with_options.return_value = client
    client.responses.create.side_effect = gpt_reply
    monkeypatch.setattr(OpenAIUtil, "get_client", lambda: client)
    monkeypatch.setattr("requests.post", slm_reply)
    monkeypatch.setattr("LLMUtil.ollama.random.choice", random.Random(0).choice)
    monkeypatch.setattr(OLLAMAUtil, "__MODEL__", model)
    monkeypatch.setattr(OLLAMAUtil, "__MODE__", "gen")
    Config.apply(config)
    inputs = dataset.example_inputs(config.num_examples)
    outputs = dataset.example_targets(config.num_examples)
    examples = inputs | outputs
    costs = CostTracker()
    with cost_session(costs):
        compositions = OpenAIUtil_Prompt.get_split_prompt(
            examples, dataset.path.name, list(inputs), list(outputs),
        )
        remaining = dataset.test_sources(config.num_examples)
        if model.startswith("gpt"):
            types = sorted({label for i in inputs for label in compositions[i]})
            labels = OpenAIUtil_Prompt.get_labels_for_types(remaining, dataset.path.name, types)
        else:
            pool = [(text, "".join(compositions[i])) for i, text in examples.items()]
            types = sorted({label for labels in compositions.values() for label in labels})
            labels = OLLAMAUtil.get_labels_for_types(remaining, dataset.path.name, types, pool)
        compositions.update(labels)
        types = sorted({RParser.key_normalize_with_underscore(label.strip("[]"))
                        for labels in compositions.values() for label in labels})
        query_ids = [i for r in planned if r.phase == "exact extraction" for i in r.query_ids]
        query = {i: dataset.text_for_id(i) for i in query_ids}
        extractor = OLLAMAUtil if local_split else OpenAIUtil
        extractor.get_split(query, dataset.path.name, list(inputs), list(outputs),
                            use_type=True, type_list=types)
    assert list(pending) == []
    assert observed == [(r.model, r.input_texts) for r in planned]
    estimate = cost_est.estimate_dataset(dataset, config)
    for key, value in costs.result_fields().items():
        assert getattr(estimate, key) == value


def test_cli_ignores_shortcuts_and_preserves_reports(corpus):
    dataset, config = corpus
    output = config.output_directory
    output.mkdir()
    for name in ("summary.csv", "pricing.csv", "Accuracy-1.csv", "errors.txt"):
        (output / name).write_text("backup")
    args = ["--data-dir", str(config.data_directory), "-pd", str(config.perfect_directory),
            "--output-dir", str(output), "--dataset", dataset.path.name,
            "-llm", "GPT5.4", "-slm", "GPT5.4", "-slm", "qwen3.5-4b",
            "--example-counts", "1,2", "-lb"]
    assert cost_est.main(args) == 0
    baseline = (output / "cost_est.csv").read_bytes()
    assert cost_est.main(args + ["-ms", "-c", "-s", "-pp", "-ps", "-pg",
                                "--num-retry", "999", "--timeout", "0", "--fail-fast"]) == 0
    assert (output / "cost_est.csv").read_bytes() == baseline
    assert cost_est.main(args + ["--qif", "100"]) == 0
    assert (output / "cost_est.csv").read_bytes() == baseline
    with (output / "cost_est.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == ["Experiment", "CF-LLM", "CF-SLM", "CF-Total"]
    assert [row["Experiment"] for row in rows] == [
        "GPT5.4_GPT5.4_1", "GPT5.4_qwen3.5-4b_1",
        "GPT5.4_GPT5.4_2", "GPT5.4_qwen3.5-4b_2",
    ]
    for name in ("summary.csv", "pricing.csv", "Accuracy-1.csv", "errors.txt"):
        assert (output / name).read_text() == "backup"
    assert set(p.name for p in output.iterdir()) == {
        "summary.csv", "pricing.csv", "Accuracy-1.csv", "errors.txt", "cost_est.csv",
    }


def test_repetitions_and_modes_sum_uncached_costs(corpus):
    _, config = corpus
    args = ["--data-dir", str(config.data_directory), "-pd", str(config.perfect_directory),
            "--output-dir", str(config.output_directory), "-n", "2", "-llm", "GPT5.4",
            "-slm", "GPT5.4"]

    def totals(extra):
        cost_est.main(args + extra)
        with (config.output_directory / "cost_est.csv").open() as handle:
            row = next(csv.DictReader(handle))
        return float(row["CF-Total"])

    assert totals(["--repetitions", "3", "--mode", "gen", "--mode", "chat"]) == 6 * totals([])


def test_default_output_is_cost_est_directly_in_results(corpus, monkeypatch, tmp_path):
    _, config = corpus
    monkeypatch.chdir(tmp_path)
    cost_est.main(["--data-dir", str(config.data_directory),
                   "-pd", str(config.perfect_directory), "-n", "2"])
    assert [p.name for p in (tmp_path / "results").iterdir()] == ["cost_est.csv"]


def test_numeric_shortcut_and_no_held_out_sources(corpus):
    dataset, config = corpus
    for length in (5, 8):
        path = dataset.path.parent / "Numbers_0.csv"
        with path.open("w", newline="") as handle:
            csv.writer(handle).writerows((str(i), str(i)) for i in range(length))
        numeric = PairDataset.from_csv(path)
        requests = requests_for(numeric, config)
        assert [r.phase for r in requests] == ["example compositions", "exact extraction"]
        assert all(r.model == "gpt-5.4" for r in requests)


def test_other_experiment_modes_do_not_run_experiments(corpus):
    dataset, config = corpus
    for experiment, expected in [("cluster", ["example compositions", "source compositions"]),
                                 ("direct-llm", ["direct transformation"])]:
        truth = cost_est.EstimationTruth(dataset, config.perfect_directory, config.num_examples)
        requests = list(cost_est.plan_requests(dataset, config, truth, experiment=experiment))
        assert [r.phase for r in requests] == expected
    direct = replace(config, num_examples=0)
    truth = cost_est.EstimationTruth(dataset, direct.perfect_directory, direct.num_examples)
    requests = list(cost_est.plan_requests(dataset, direct, truth, experiment="direct-llm"))
    assert requests[0].query_ids == (5, 6, 7)
    assert requests[0].response == "Jones, Carol\nWHITE, DAN\nGREEN, EVE"


def test_nonsemantic_clustering_counts_extraction_finalization(corpus):
    dataset, config = corpus
    requests = requests_for(dataset, replace(config, cluster_config="collapse"))
    assert [r.phase for r in requests] == ["exact extraction", "final extraction"]
    assert requests[0].query_ids == requests[1].query_ids
