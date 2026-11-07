import csv
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import benchmarkgpt
import cost_est
from gdt.cli import main, parse_arguments
from gdt.config import Config, ExperimentConfig
from gdt.direct_benchmark import _predict, run_direct_benchmark
from gdt.direct_models import UnsupportedModelError, check_models
from LLMUtil.providers import NovitaUtil, OpenAIUtil


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("test attempted a live provider request")
    monkeypatch.setattr(OpenAIUtil, "get_client", forbidden)
    monkeypatch.setattr(NovitaUtil, "get_client", forbidden)
    yield
    Config.apply(ExperimentConfig())


def client_with_models(monkeypatch, provider, models):
    client = Mock()
    client.with_options.return_value = client
    client.models.list.return_value = [SimpleNamespace(id=model) for model in models]
    adapter = OpenAIUtil if provider == "openai" else NovitaUtil
    monkeypatch.setattr(adapter, "get_client", lambda: client)
    return client


def write_datasets(tmp_path):
    data = tmp_path / "data_shuffled"
    data.mkdir()
    for name in ("A.csv", "B.csv"):
        (data / name).write_text("Alice,A\n" * 6)
    return data


@pytest.mark.parametrize("extra", [
    ["-slm", "qwen3-4b"], ["--output-dir", "elsewhere"], ["--data-dir", "elsewhere"],
    ["--dataset", "A.csv"], ["--example-counts", "1,2"], ["--repetitions", "2"],
    ["-ms"], ["-pg"], ["-ps"], ["-pp"], ["-lb"], ["-sp"], ["-c"], ["-s"],
    ["--num-retry", "3"], ["--timeout", "10"], ["--mode", "chat"], ["--fail-fast"],
    ["--num-examples", "1"], ["--provider-extra", "openai"], ["--prov", "openai"],
    ["-ll", "GPT5.4"], ["extra"], ["--unknown"], ["-slm", "x", "--help"],
    ["--provider", "-slm", "x"],
])
def test_direct_rejects_other_arguments_before_work(extra, monkeypatch, capsys, tmp_path):
    run = Mock()
    monkeypatch.setattr("gdt.direct_benchmark.run_direct_benchmark", run)
    Config.model = "unchanged"
    with pytest.raises(SystemExit) as caught:
        main(["--experiment", "direct-llm", "--provider", "openai",
              "-llm", "GPT5.4", "-n", "1,2,3,4,5", *extra])
    assert caught.value.code == 2
    message = capsys.readouterr()
    assert message.out == ""
    assert message.err == (
        "Some of the provided arguments are not supported for the direct LLM experiment.\n"
        "The valid, full command is: python main.py --experiment direct-llm "
        "--provider Provider_Name -llm LLM_Name -n Number_of_Examples\n"
    )
    assert Config.model == "unchanged"
    run.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_llm_selects_actual_model_and_metadata(monkeypatch):
    run = Mock(return_value=[])
    monkeypatch.setattr("gdt.direct_benchmark.run_direct_benchmark", run)
    assert main(["--experiment=direct-llm", "--provider=openai", "-llm=GPT5.4", "-n2"]) == 0
    config = run.call_args.args[0]
    assert config.model == config.llm_model == "gpt-5.4"
    assert config.num_examples == 2
    assert run.call_args.kwargs == {"provider": "openai", "example_counts": [2]}
    args = parse_arguments(["--experiment", "direct-llm", "--provider", "novita"])
    assert args.config.llm_model == "deepseek"
    assert args.config.num_examples == 5
    assert args.example_counts == [5]


@pytest.mark.parametrize("arguments", [
    ["-n", "1,2,3,4,5"], ["-n=1,2,3,4,5"], ["-n1,2,3,4,5"],
])
def test_direct_count_list_reaches_benchmark(arguments, monkeypatch):
    run = Mock(return_value=[])
    monkeypatch.setattr("gdt.direct_benchmark.run_direct_benchmark", run)
    assert main(["--experiment", "direct-llm", "--provider", "openai", "-llm", "GPT5.4",
                 *arguments]) == 0
    assert run.call_args.args[0].num_examples == 1
    assert run.call_args.kwargs == {"provider": "openai", "example_counts": [1, 2, 3, 4, 5]}


@pytest.mark.parametrize("value", ["6", "1,6", "1,-1", "1,two", ""])
def test_invalid_count_in_list_stops_before_work(value, monkeypatch, tmp_path):
    run = Mock()
    monkeypatch.setattr("gdt.direct_benchmark.run_direct_benchmark", run)
    with pytest.raises(SystemExit) as caught:
        main(["--experiment", "direct-llm", "-n", value])
    assert caught.value.code == 2
    run.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_direct_runs_and_summarizes_every_count(monkeypatch, tmp_path):
    write_datasets(tmp_path)
    client = client_with_models(monkeypatch, "openai", ["gpt-5.4"])
    client.responses.create.return_value = SimpleNamespace(output_text="A")
    assert main(["--experiment", "direct-llm", "--provider", "openai", "-llm", "GPT5.4",
                 "-n", "1,2,3,4,5"]) == 0
    client.models.list.assert_called_once_with()
    calls = client.responses.create.call_args_list
    counts = [n for n in range(1, 6) for _ in range(2)]
    assert len(calls) == len(counts)
    for call, count in zip(calls, counts):
        assert call.kwargs["input"] == "Q: Alice A: A\n" * count + "Q: Alice A:"
    output = tmp_path / "results" / "direct_llm"
    with (output / "summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["num_examples"]) for row in rows] == counts
    assert all(float(row["accuracy"]) == 1 for row in rows)
    with (output / "pricing.csv").open() as handle:
        pricing = list(csv.DictReader(handle))
    assert [row["Experiment"] for row in pricing] == [
        f"GPT5.4_GPT5.4_{n}" for n in range(1, 6)
    ]


def test_standalone_entry_reads_real_command_line(monkeypatch, capsys):
    run = Mock(return_value=[])
    monkeypatch.setattr("gdt.direct_benchmark.run_direct_benchmark", run)
    monkeypatch.setattr("sys.argv", ["benchmarkgpt.py", "--provider", "novita", "-llm", "glm",
                                    "-n", "0,3"])
    assert benchmarkgpt.benchmark() == 0
    assert run.call_args.args[0].llm_model == "glm"
    assert run.call_args.args[0].num_examples == 0
    assert run.call_args.kwargs == {"provider": "novita", "example_counts": [0, 3]}
    with pytest.raises(SystemExit):
        benchmarkgpt.benchmark(["-slm", "ignored"])
    assert "python benchmarkgpt.py --provider Provider_Name -llm LLM_Name -n Number_of_Examples" \
        in capsys.readouterr().err


@pytest.mark.parametrize("provider,alias,api_model", [
    ("openai", "GPT5.4", "gpt-5.4"), ("openai", "GPT6", "gpt-6-astra"),
    ("openai", "custom-future-model", "custom-future-model"),
    ("novita", "deepseek", "deepseek/deepseek-v3.2"),
    ("novita", "glm", "zai-org/glm-4.7"),
    ("novita", "custom/model", "custom/model"),
])
def test_provider_catalog_uses_aliases_without_a_fixed_allowlist(monkeypatch, provider, alias,
                                                               api_model):
    client = client_with_models(monkeypatch, provider, [api_model])
    check_models([alias], provider)
    client.models.list.assert_called_once_with()
    client.with_options.assert_called_once_with(max_retries=0, timeout=30)
    client.responses.create.assert_not_called()
    client.completions.create.assert_not_called()


@pytest.mark.parametrize("provider", ["openai", "novita"])
def test_unsupported_model_stops_before_any_work(provider, monkeypatch, capsys, tmp_path):
    client_with_models(monkeypatch, provider, ["a-supported-model"])
    output = tmp_path / "results"
    output.mkdir()
    for name in ("errors.txt", "progress.log", "summary.csv"):
        (output / name).write_text("backup")
    predict = Mock()
    monkeypatch.setattr("gdt.direct_benchmark._predict", predict)
    assert main(["--experiment", "direct-llm", "--provider", provider, "-llm", "missing-model",
                 "-n", "1"]) == 1
    message = capsys.readouterr()
    assert message.out == ""
    assert message.err == f"Model 'missing-model' is not supported by provider '{provider}'.\n"
    predict.assert_not_called()
    assert len(list(output.iterdir())) == 3
    assert all(path.read_text() == "backup" for path in output.iterdir())


def test_validation_failure_is_not_mislabeled_as_unsupported(monkeypatch, capsys, tmp_path):
    client = client_with_models(monkeypatch, "openai", [])
    client.models.list.side_effect = ConnectionError("offline")
    assert main(["--experiment", "direct-llm", "-llm", "GPT5.4"]) == 1
    message = capsys.readouterr().err
    assert "Could not check models" in message
    assert "not supported" not in message
    assert not (tmp_path / "results").exists()


def test_openai_prediction_uses_llm_not_slm(monkeypatch, tmp_path):
    from gdt.data import PairDataset

    data = write_datasets(tmp_path)
    send = Mock(return_value="A")
    monkeypatch.setattr(OpenAIUtil, "create_response", send)
    config = ExperimentConfig(llm_model="GPT5.4", model="qwen3.5-4b", num_examples=1)
    assert _predict(PairDataset.from_csv(data / "A.csv"), config, "openai") == ["A"]
    assert send.call_args.kwargs["model"] == "gpt-5.4"
    assert send.call_args.kwargs["num_retry"] == 1


class ProviderRejection(Exception):
    def __init__(self, code, message, status=400):
        super().__init__(message)
        self.body = {"error": {"code": code, "message": message}}
        self.status_code = status


@pytest.mark.parametrize("provider,model", [("openai", "gpt-5.4"), ("novita", "custom/model")])
@pytest.mark.parametrize("error", [
    ProviderRejection("model_not_found", "model is absent", 404),
    ProviderRejection(None, "This model is not supported in this endpoint"),
    ProviderRejection(None, "You do not have access to this model", 403),
    ProviderRejection(None, "This model does not support this endpoint"),
])
def test_inference_model_rejection_stops_all_datasets(provider, model, error, monkeypatch,
                                                    capsys, tmp_path):
    write_datasets(tmp_path)
    client = client_with_models(monkeypatch, provider, [model])
    client.responses.create.side_effect = error
    client.completions.create.side_effect = error
    assert main(["--experiment", "direct-llm", "--provider", provider, "-llm", model, "-n", "1"]) \
        == 1
    assert "not supported" in capsys.readouterr().err
    send = client.responses.create if provider == "openai" else client.completions.create
    send.assert_called_once()
    progress = (tmp_path / "results" / "progress.log").read_text()
    assert 'dataset="A.csv"' in progress
    assert 'dataset="B.csv"' not in progress


def test_late_model_rejection_preserves_completed_results(monkeypatch, tmp_path):
    data = write_datasets(tmp_path)
    client_with_models(monkeypatch, "openai", ["gpt-5.4"])
    predict = Mock(side_effect=[["A"], ProviderRejection("unsupported_model", "rejected")])
    monkeypatch.setattr("gdt.direct_benchmark._predict", predict)
    config = ExperimentConfig(llm_model="gpt-5.4", data_directory=data,
                              output_directory=tmp_path / "results", num_examples=1)
    with pytest.raises(UnsupportedModelError):
        run_direct_benchmark(config)
    with (config.output_directory / "direct_llm" / "summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["dataset"] for row in rows] == ["A.csv", "B.csv"]
    assert rows[0]["model"] == rows[0]["llm_model"] == "gpt-5.4"
    assert rows[0]["error"] == ""
    assert "not supported" in rows[1]["error"]


def test_ordinary_dataset_failure_can_still_continue(monkeypatch, tmp_path):
    data = write_datasets(tmp_path)
    client_with_models(monkeypatch, "openai", ["gpt-5.4"])
    predict = Mock(side_effect=[ProviderRejection("invalid_api_key", "authentication failed", 401),
                                ["A"]])
    monkeypatch.setattr("gdt.direct_benchmark._predict", predict)
    results = run_direct_benchmark(ExperimentConfig(
        llm_model="gpt-5.4", data_directory=data,
        output_directory=tmp_path / "results", num_examples=1,
    ))
    assert [result.succeeded for result in results] == [False, True]
    assert "not supported" not in results[0].error


@pytest.mark.parametrize("counts", [[1], [0, 1, 2, 3, 4, 5]])
def test_cost_estimation_uses_direct_llm_argument_without_provider_access(tmp_path, counts):
    write_datasets(tmp_path)
    assert cost_est.main(["--experiment", "direct-llm", "--provider", "openai", "-llm", "GPT5.4",
                          "-n", ",".join(map(str, counts))]) == 0
    with (tmp_path / "results" / "cost_est.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["Experiment"] for row in rows] == [f"GPT5.4_GPT5.4_{n}" for n in counts]
    assert all(float(row["CF-LLM"]) > 0 for row in rows)
