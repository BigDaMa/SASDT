from __future__ import annotations

from pathlib import Path

from gdt.diagnostics import dataset_diagnostics, split_phase, write_dataset_error
from gdt.timing import PipelineTimer, stop_pipeline_timer
from gdt.timeout import dataset_time_limit
from gdt.config import Config, ExperimentConfig
from gdt.data import PairDataset
from gdt.ground_truth import ground_truth_session
from gdt.results import EvaluationResult, RunResult
from gdt.progress import progress_phase
from gdt.pricing import CostTracker, cost_session
from gdt.models import is_gpt_model


class PipelineRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        config.validate()
        self.config = config

    def run(self, dataset_path: str | Path) -> RunResult:
        from LLMUtil.ollama import OLLAMAUtil

        path = Path(dataset_path)
        timer = PipelineTimer()
        costs = CostTracker()
        try:
            with (
                dataset_diagnostics(self.config.output_directory, path.name, self.config) as diagnostics,
                ground_truth_session(),
                OLLAMAUtil.dataset_session(),
                cost_session(costs),
                timer,
                dataset_time_limit(self.config.timeout_seconds),
            ):
                evaluation = self._run(path)
            return RunResult(
                dataset=path.name,
                model=self.config.model,
                llm_model=self.config.llm_model,
                mode=self.config.mode,
                num_examples=self.config.num_examples,
                duration_seconds=timer.elapsed,
                evaluation=evaluation,
                **diagnostics.result_fields(),
                **costs.result_fields(),
            )
        except Exception as error:
            write_dataset_error(self.config.output_directory, path.name, self.config, error)
            return RunResult(
                dataset=path.name,
                model=self.config.model,
                llm_model=self.config.llm_model,
                mode=self.config.mode,
                num_examples=self.config.num_examples,
                duration_seconds=timer.elapsed,
                error=f"{type(error).__name__}: {error}",
                **diagnostics.result_fields(),
                **costs.result_fields(),
            )

    def _run(self, path: Path) -> EvaluationResult:
        # Imports are deliberately local: `--help` and data-only tools should not
        # import torch, transformers, clustering models, or OpenAI clients.
        from Clustering.clustermeta import ClusterMeta
        from DataAccess.dataaccess import DataAccess
        from DataAccess.datafeature import DataFeature
        from LLMUtil.LLMUtil import OLLAMAUtil, OpenAIUtil
        from LLMUtil.ResponseParser import RParser
        from Propagate.propagate import Propagate
        from Propagate.witness import get_cluster_operators
        from transform.Evaluation import Evaluation
        from transform.Transform import Transform
        from Utils.IterPrintUtil import IterPrintUtil

        progress_phase("data loading")
        dataset = PairDataset.from_csv(path)
        dataset.validate_example_count(self.config.num_examples)
        if self.config.num_examples == 0:
            raise ValueError("the full GDT pipeline requires at least one example")
        Config.apply(self.config)
        Config.data_directory = str(path.parent)
        DataAccess.DATA_FOLDER = str(path.parent)
        DataAccess.NUM_EXAMPLES = self.config.num_examples
        DataFeature.clear()
        OLLAMAUtil.set_model(self.config.model)
        OLLAMAUtil.set_mode(self.config.mode)

        example_input = dataset.example_inputs(self.config.num_examples)
        example_output = dataset.example_targets(self.config.num_examples)
        example_strings = example_input | example_output
        clustering_values = dataset.clustering_mapping(self.config.num_examples)

        with split_phase("semantic type discovery"):
            split_clusters = ClusterMeta.run_cluster(
                path.name,
                method="collapse",
                clusteralg="DBSCAN",
                config=self.config.cluster_config,
            )
        mapped_clusters = DataAccess.map_clustered_idx_to_str(split_clusters, clustering_values)
        mapped_clusters = [dict(sorted(cluster.items())) for cluster in mapped_clusters]

        query_ids = self._select_semantic_split_ids(
            split_clusters,
            mapped_clusters,
            self.config.num_examples,
            output_ids=set(example_output),
        )
        query = {value_id: clustering_values[value_id] for value_id in query_ids}
        semantic_types = sorted(
            {
                RParser.key_normalize_with_underscore(item)
                for item in DataFeature.get_all_sem_types(path.name)
            }
        )
        extractor = (
            OLLAMAUtil
            if self.config.slm_split and not is_gpt_model(self.config.model)
            else OpenAIUtil
        )
        with split_phase("semantic component extraction"):
            semantic_components = extractor.get_split(
                query,
                path.name,
                list(example_input),
                list(example_output),
                use_type=bool(semantic_types),
                type_list=semantic_types,
            )
        if semantic_components is None:
            raise RuntimeError("semantic component parsing returned no result")

        query_strings = list(query.values())
        query_ids_ordered = list(query)
        progress_phase("position inference")
        valid_positions = get_cluster_operators(
            mapped_clusters,
            semantic_components,
            query_strings,
            query_ids_ordered,
            path.name,
            output_ids=set(example_output),
        )
        progress_phase("propagation")
        propagated = Propagate.propagate(mapped_clusters, valid_positions, semantic_components)

        input_components = {
            value_id: propagated[value_id]
            for value_id in example_input
            if value_id in propagated
        }
        output_components = {
            dataset.row_id_from_target(value_id): propagated[value_id]
            for value_id in example_output
            if value_id in propagated
        }
        test_components = {
            value_id: {
                "id": value_id,
                "text": dataset.text_for_id(value_id),
                "components": propagated[value_id],
            }
            for value_id in dataset.test_sources(self.config.num_examples)
            if value_id in propagated and value_id not in example_strings
        }

        progress_phase("transformation synthesis")
        transformations, transformation_function = Transform.transform_with_function(
            example_input,
            input_components,
            example_output,
            output_components,
            test_components,
            dataset.examples(self.config.num_examples),
        )
        stop_pipeline_timer()
        progress_phase("evaluation")
        evaluation = Evaluation.evaluate(
            transformations,
            dataset.ground_truth(self.config.num_examples),
            dataset.test_sources(self.config.num_examples),
        )
        print("Transformation function:")
        IterPrintUtil.print(
            Transform.describe_transformation_function(transformation_function)
        )
        Evaluation.print_eval_result(evaluation)
        return EvaluationResult.from_mapping(evaluation)

    @staticmethod
    def _select_semantic_split_ids(
        split_clusters: list[set[int]],
        mapped_clusters: list[dict[int, str]],
        num_examples: int,
        *,
        output_ids: set[int] | None = None,
    ) -> list[int]:
        if not split_clusters:
            raise ValueError("clustering returned no clusters")

        if output_ids is None:
            output_ids = set(split_clusters[-1])
        selected: list[int] = []
        for cluster in mapped_clusters:
            input_values = [value_id for value_id in cluster if value_id not in output_ids]
            if not input_values:
                continue
            # Prefer demonstration inputs.  If a cluster has none, use its stable
            # first member to provide one semantic-split witness.
            demonstration_ids = [
                value_id for value_id in input_values if value_id < num_examples
            ]
            candidates = demonstration_ids or [min(input_values)]
            for value_id in candidates:
                if value_id not in selected:
                    selected.append(value_id)
        for value_id in sorted(output_ids):
            if value_id not in selected:
                selected.append(value_id)
        return selected
