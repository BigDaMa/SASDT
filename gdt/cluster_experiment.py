from __future__ import annotations

from typing import Sequence

from gdt.diagnostics import clear_errors, dataset_diagnostics, split_phase, write_dataset_error
from gdt.timing import PipelineTimer, stop_pipeline_timer
from gdt.timeout import dataset_time_limit
from gdt.config import Config, ExperimentConfig
from gdt.data import PairDataset, discover_csv_files
from gdt.ground_truth import GroundTruth, ground_truth_session, perfect_path
from gdt.output import dataset_output
from gdt.progress import ProgressLog, progress_phase
from gdt.results import EvaluationResult, RunResult
from gdt.pricing import CostTracker, cost_session


def run_cluster_experiment(
    config: ExperimentConfig,
    *,
    datasets: Sequence[str] | None = None,
) -> list[RunResult]:
    from Clustering.clustereval import ClusterEvaluator
    from Clustering.clustermeta import ClusterMeta
    from DataAccess.dataaccess import DataAccess
    from DataAccess.datafeature import DataFeature
    from LLMUtil.ollama import OLLAMAUtil

    clear_errors(config.output_directory)
    Config.apply(config)
    DataAccess.DATA_FOLDER = str(config.data_directory)
    DataAccess.NUM_EXAMPLES = config.num_examples
    results = []
    paths = discover_csv_files(config.data_directory, datasets)
    progress = ProgressLog(config.output_directory, "cluster", len(paths))
    for path in paths:
        with dataset_output(path.name), progress.run(path.name, config) as current:
            timer = PipelineTimer()
            costs = CostTracker()
            try:
                with (
                    dataset_diagnostics(config.output_directory, path.name, config) as diagnostics,
                    ground_truth_session(),
                    OLLAMAUtil.dataset_session(),
                    cost_session(costs),
                    timer,
                    dataset_time_limit(config.timeout_seconds),
                    split_phase("semantic type discovery"),
                ):
                    dataset = PairDataset.from_csv(path)
                    dataset.validate_example_count(config.num_examples)
                    DataFeature.clear()
                    clusters = ClusterMeta.run_cluster(path.name, config=config.cluster_config)
                    stop_pipeline_timer()
                    progress_phase("cluster evaluation")
                    truth_path = perfect_path(path.name, "_cluster.txt", config.perfect_directory)
                    metrics: dict[str, float] = {"cluster_count": float(len(clusters))}
                    if truth_path.is_file():
                        universe = set().union(*clusters) if clusters else set()
                        truth = [
                            restricted
                            for group in GroundTruth(dataset, config.perfect_directory,
                                                     config.num_examples).clusters
                            if (restricted := group & universe)
                        ]
                        precision, recall, f1 = ClusterEvaluator.evaluate_pairwise_f1(
                            truth, clusters
                        )
                        metrics.update(precision=precision, recall=recall, f1=f1)
                    else:
                        f1 = 0.0
                    result = RunResult(
                        dataset=path.name,
                        model=config.model,
                        llm_model=config.llm_model,
                        mode="cluster",
                        num_examples=config.num_examples,
                        duration_seconds=timer.elapsed,
                        **diagnostics.result_fields(),
                        **costs.result_fields(),
                        evaluation=EvaluationResult(
                            accuracy=f1,
                            correct=0,
                            incorrect=0,
                            incorrect_items=[metrics],
                        ),
                    )
                    print(f"{path.name}: clusters={len(clusters)}, pairwise_f1={f1:.4f}")
            except Exception as error:
                write_dataset_error(config.output_directory, path.name, config, error)
                result = RunResult(
                    dataset=path.name,
                    model=config.model,
                    llm_model=config.llm_model,
                    mode="cluster",
                    num_examples=config.num_examples,
                    duration_seconds=timer.elapsed,
                    error=f"{type(error).__name__}: {error}",
                    **diagnostics.result_fields(),
                    **costs.result_fields(),
                )
                print(f"{path.name}: ERROR: {result.error}")
            current.finish(result)
        results.append(result)
    return results
