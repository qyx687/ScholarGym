#!/usr/bin/env python3
import os
import json
import shutil
import importlib.util
from typing import List, Dict
from tqdm import tqdm
import numpy as np
import datetime
import hashlib
from langchain_ollama import OllamaEmbeddings

from logger import get_logger
from rag import CitationRAGSystem
import config
from deeprag import DeepResearchWorkflow
from simplerag import SimpleWorkflow
from utils import extract_ground_truth_arxiv_ids, CheckpointManager, calculate_retrieval_metrics, AgentTraceRecorder
from deep_retrieval import DeepRetrievalProcessor
from graph_methods import (
    ArtifactWriter,
    BoundedEmbeddingProvider,
    DEFAULT_FEATURE_WEIGHTS,
    INTENT_WEIGHTS,
    PAPER_EMBEDDING_SERIALIZATION_ID,
    PerSubqueryProcessor,
    QUERY_SCOPED_EMBEDDING_CACHE_POLICY,
    QueryScopedEmbeddingCache,
    RERANK_FORMULA_ID,
    S2GraphClient,
    load_paper_db,
)
from onepass_postprocess import OnePassPostprocessor, aggregate_postprocess_metrics
from dimension_catalog import CATALOG_VERSION, POLICY_VERSION, PROMPT_VERSION
from online_paper_type import S2PublicationTypeResolver
from rerank_skill import (
    DEFAULT_MAX_NEGATIVE_MASS,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_NEGATIVE_WEIGHT,
    DEFAULT_SEMANTIC_MIN_MASS,
    PAPER_TYPE_BACKEND,
    S2_NATIVE_PAPER_TYPE_NAMESPACE,
    RerankSkill,
)

logger = get_logger(__name__, log_file='./log/eval.log')


def package_source_sha256(filenames):
    digest = hashlib.sha256()
    base = os.path.dirname(os.path.abspath(__file__))
    for filename in sorted(filenames):
        path = os.path.join(base, filename)
        digest.update(filename.encode('utf-8') + b'\0')
        with open(path, 'rb') as handle:
            digest.update(handle.read())
    return digest.hexdigest()


def file_identity(path: str, *, include_sha256: bool = False) -> Dict:
    resolved = os.path.realpath(path)
    identity: Dict = {"path": resolved, "exists": os.path.isfile(resolved)}
    if not identity["exists"]:
        return identity
    stat = os.stat(resolved)
    identity.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    if include_sha256:
        digest = hashlib.sha256()
        with open(resolved, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        identity["sha256"] = digest.hexdigest()
    return identity


def signature_sha256(payload: Dict) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_resume_signature(
    manifest_path: str,
    detailed_results_path: str,
    expected_signature: str,
    *,
    expected_payload: Dict = None,
    allow_compatible_code_change: bool = False,
) -> Dict:
    artifacts_dir = os.path.dirname(manifest_path)
    has_checkpoint = os.path.isfile(detailed_results_path) and os.path.getsize(detailed_results_path) > 0
    has_artifact_rows = any(
        filename.endswith(".jsonl") and os.path.getsize(os.path.join(root, filename)) > 0
        for root, _, filenames in os.walk(artifacts_dir)
        if ".staging" not in root.split(os.sep)
        for filename in filenames
    )
    if not (has_checkpoint or has_artifact_rows):
        return {"mode": "new"}
    if not os.path.isfile(manifest_path):
        raise ValueError(
            "Cannot safely resume: checkpoint/artifact rows exist but the run "
            f"manifest is missing: {manifest_path}. Use a new --run_label (or an empty output directory)."
        )
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
    except Exception as exc:
        raise ValueError(
            f"Cannot safely resume: existing run manifest is unreadable: {manifest_path}"
        ) from exc
    actual_signature = existing.get("run_signature_sha256")
    if actual_signature == expected_signature:
        return {"mode": "exact", "previous_run_signature_sha256": actual_signature}
    if allow_compatible_code_change and expected_payload is not None:
        actual_payload = existing.get("run_signature")
        if isinstance(actual_payload, dict):
            actual_semantics = dict(actual_payload)
            expected_semantics = dict(expected_payload)
            # This opt-in is intended for result-preserving implementation
            # upgrades (such as bounded parallel execution). All recorded
            # data/method settings must remain byte-for-byte equivalent.
            compatible_upgrade_keys = {
                "package_source_sha256",
                "postprocess_event_workers",
                "postprocess_selector_concurrency",
                "postprocess_embedding_concurrency",
                "postprocess_embedding_cache_policy",
            }
            for key in compatible_upgrade_keys:
                actual_semantics.pop(key, None)
                expected_semantics.pop(key, None)
            if actual_semantics == expected_semantics:
                return {
                    "mode": "compatible_code_change",
                    "previous_run_signature_sha256": actual_signature,
                }
    raise ValueError(
        "Refusing to resume into an output directory created by a different "
        "code/data/method configuration. Use a new --run_label, or use "
        "--allow_resume_compatible_code_change only for an audited result-preserving code upgrade."
    )

def load_config_from_path(config_path: str):
    """
    Dynamically load config from a file path.
    
    Args:
        config_path: Path to config.py file
        
    Returns:
        Loaded config module
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    spec = importlib.util.spec_from_file_location("config_custom", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    logger.info(f"[📝] Loaded config from: {config_path}")
    return config_module

class CitationEvaluator:
    def __init__(self, rag_system: CitationRAGSystem, llm_model: str = config.LLM_MODEL_NAME, 
                 is_local: bool = config.IS_LOCAL_LLM, prompt_type: str = config.EVAL_PROMPT_TYPE, 
                 search_method: str = config.EVAL_SEARCH_METHOD, trace_recorder=None, onepass_postprocessor=None):
        self.rag_system = rag_system
        self.llm_model = llm_model
        self.is_local = is_local
        self.prompt_type = prompt_type
        self.search_method = search_method
        self.gen_params = config.LLM_GEN_PARAMS
        self.trace_recorder = trace_recorder
        
        # Validate search method
        available_methods = self.rag_system.get_available_search_methods()
        if search_method not in available_methods:
            raise ValueError(f"Search method '{search_method}' not available. Available methods: {available_methods}")
        
        # Initialize workflows
        self.simple_workflow = SimpleWorkflow(
            rag_system=self.rag_system,
            llm_model=self.llm_model,
            gen_params=self.gen_params,
            is_local=self.is_local,
            prompt_type=self.prompt_type
        )
        
        self.deep_research_workflow = DeepResearchWorkflow(
            rag_system=self.rag_system,
            llm_model=self.llm_model,
            gen_params=self.gen_params,
            is_local=self.is_local,
            trace_recorder=trace_recorder,
            onepass_postprocessor=onepass_postprocessor,
        )

    def load_benchmark_data(self, benchmark_jsonl_path: str) -> List[Dict]:
        """Load benchmark data from JSONL file."""
        benchmark_data = []
        with open(benchmark_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                benchmark_data.append(json.loads(line.strip()))
        
        logger.info(f"[📂]Loaded {len(benchmark_data)} benchmark queries")
        return benchmark_data

    def evaluate_single_query_deep_research(
        self, 
        query_data: Dict, 
        results_per_query: int = None, 
        max_iterations: int = 3, 
        idx: int = 1
    ) -> Dict:
        """
        Evaluate a single benchmark query using the Deep Research workflow.
        
        Args:
            query_data: Query data dictionary
            results_per_query: Number of results per retrieval (defaults to config.MAX_RESULTS_PER_QUERY)
            max_iterations: Maximum iterations for deep research
            idx: Query index for logging
        """
        query = query_data['query']
        gt_labels = query_data['gt_label']
        gt_arxiv_ids = extract_ground_truth_arxiv_ids(query_data['cited_paper'], gt_labels)

        postprocessor = self.deep_research_workflow.onepass_postprocessor
        materialize_without_ground_truth = (
            postprocessor is not None
            and postprocessor.postprocess_stage == 'materialize'
        )
        if not gt_arxiv_ids and not materialize_without_ground_truth:
            return None
        if not gt_arxiv_ids:
            logger.warning(
                f"[⚠️] Query {idx} has no positive arXiv ground truth; "
                "Stage A will materialize its pools but exclude it from pool-recall aggregates"
            )

        workflow_results = self.deep_research_workflow.run(
            query_data, 
            gt_arxiv_ids=gt_arxiv_ids,
            results_per_query=results_per_query,
            max_iterations=max_iterations,
            idx=idx,
        )
        
        # TODO[fix]: Handle workflow failure, early stop
        if workflow_results is None:
            logger.warning(f"[❌] Query {idx} failed - workflow returned None")
            return None
        
        iteration_results = []
        # Use 'select' stage to compute per-iteration metrics
        select_steps = (
            [
                item
                for item in workflow_results['history']
                if item.get('stage') == 'select'
            ]
            if gt_arxiv_ids
            else []
        )

        # Track cumulative retrieval and selection across iterations
        all_retrieved_arxiv_ids = set()
        all_selected_arxiv_ids = set()
        total_discarded_gt_count = 0  # Cumulative count of discarded ground truth papers

        for step in select_steps:
            iter_idx = step.get('iter_idx')
            retrieved_in_iter = step.get('retrieved_papers') or {}
            selected_in_iter = step.get('selected_papers') or {}
            gt_rank = step.get('gt_rank') or []
            browsing_arxiv_ids = step.get('browsing_arxiv_ids') or []
            avg_distance = step.get('avg_distance', 0)
            iteration_metrics = step.get('iteration_metrics') or {}
            subquery_metrics = step.get('subquery_metrics') or {}
            
            if not isinstance(retrieved_in_iter, dict):
                retrieved_in_iter = {}
            if not isinstance(selected_in_iter, dict):
                selected_in_iter = {}

            # Collect current iteration ArXiv IDs
            current_iter_retrieved = {
                p.get('arxiv_id')
                for papers in retrieved_in_iter.values()
                for p in papers if p.get('arxiv_id')
            }
            current_iter_selected = {
                p.get('arxiv_id')
                for papers in selected_in_iter.values()
                for p in papers if p.get('arxiv_id')
            }

            # Update cumulative sets
            all_retrieved_arxiv_ids.update(current_iter_retrieved)
            all_selected_arxiv_ids.update(current_iter_selected)

            # Calculate cumulative metrics using unified function
            metrics = calculate_retrieval_metrics(
                gt_arxiv_ids, 
                all_retrieved_arxiv_ids, 
                all_selected_arxiv_ids
            )
            
            # Count discarded ground truth in this iteration
            iter_discarded_gt = iteration_metrics.get('discarded_gt_count', 0)
            total_discarded_gt_count += iter_discarded_gt

            # Calculate retrieved but not selected GTs in this iteration
            iter_retrieved_not_selected_gt = (gt_arxiv_ids & current_iter_retrieved) - current_iter_selected
            
            missed_gt_ratio = len(iter_retrieved_not_selected_gt) / len(current_iter_retrieved) if current_iter_retrieved else 0.0

            iteration_results.append({
                "iter_idx": iter_idx,
                "iter_retrieved_not_selected_gt": list(iter_retrieved_not_selected_gt),
                "missed_gt_ratio": missed_gt_ratio,
                # Selection metrics (after selector filtering)
                "recall": metrics['recall'],
                "precision": metrics['precision'],
                "matches": metrics['matches'],
                "selected_count": metrics['selected_count'],
                # Retrieval metrics (before selector filtering)
                "retrieval_recall": metrics['retrieval_recall'],
                "retrieval_precision": metrics['retrieval_precision'],
                "retrieval_matches": metrics['retrieval_matches'],
                "retrieved_count": metrics['retrieved_count'],
                # Detailed metrics
                "iteration_metrics": iteration_metrics,
                "subquery_metrics": subquery_metrics,
                "iter_discarded_gt_count": iter_discarded_gt,
                "total_discarded_gt_count": total_discarded_gt_count,
                "gt_rank": gt_rank,
                'avg_distance': avg_distance,
                "total_gt": len(gt_arxiv_ids),
                "planner_during": step.get('planner_during', -1),
                "retrieval_during": step.get('retrieval_during', -1),
                "selector_during": step.get('selector_during', -1),
                "browser_during": step.get('browser_during', -1),
                "overhead_during": step.get('overhead_during', -1),
                "total_during": step.get('total_during', -1),
                "current_iter_retrieved": list(current_iter_retrieved),
                "current_iter_selected": list(current_iter_selected),
                "current_iter_browsing": browsing_arxiv_ids,
            })
            
            # Log cumulative metrics
            logger.info(
                f"[📊 Cumulative Metrics after Iter {iter_idx}] "
                f"Retrieved: {metrics['retrieved_count']} papers, "
                f"Retrieval Recall: {metrics['retrieval_recall']:.4f}, "
                f"Retrieval Precision: {metrics['retrieval_precision']:.4f}; "
                f"Selected: {metrics['selected_count']} papers, "
                f"Selection Recall: {metrics['recall']:.4f}, "
                f"Selection Precision: {metrics['precision']:.4f}; "
                f"Total Discarded GT: {total_discarded_gt_count}"
            )

        final_selected_papers = [
            {"arxiv_id": p.arxiv_id}
            for p in workflow_results.get('selected_papers', [])
        ]

        return {
            'idx': idx,
            'query': query,
            'ground_truth_arxiv_ids': list(gt_arxiv_ids),
            'ground_truth_metrics_available': bool(gt_arxiv_ids),
            'iteration_results': iteration_results,
            'final_report': workflow_results.get('final_report', ''),
            'final_selected_papers': final_selected_papers,
            'executed_queries': workflow_results.get('executed_queries', []),
            'postprocess_results': workflow_results.get('postprocess_results', {}),
        }

    def evaluate_benchmark(
        self, 
        benchmark_data: List[Dict], 
        workflow: str = 'simple', 
        top_k_list: List[int] = None, 
        results_per_query: int = None,
        max_iterations: int = 3,
        detailed_results_file: str = None,
        enable_resume: bool = True
    ) -> Dict:
        """
        Evaluate the entire benchmark dataset.
        
        Args:
            benchmark_data: List of benchmark queries
            workflow: 'simple' or 'deep_research'
            top_k_list: Top-k values for simple workflow (ignored for deep_research)
            results_per_query: Results per query for deep_research (defaults to config.MAX_RESULTS_PER_QUERY)
            max_iterations: Maximum iterations for deep_research workflow
            detailed_results_file: Path to save detailed results incrementally (JSONL format)
            enable_resume: Enable resume from checkpoint
        """
        logger.info(f"Evaluating {len(benchmark_data)} queries with results_per_query={results_per_query} (using top_k={top_k_list} for simple workflow), max_iterations={max_iterations}")
        logger.info(f"Using prompt type: {self.prompt_type}\nUsing search method: {self.search_method}\nUsing workflow: {workflow}")

        # TODO[resume]: Initialize checkpoint manager
        checkpoint_manager = None
        artifact_reconciliation = None
        if enable_resume and detailed_results_file:
            checkpoint_manager = CheckpointManager(detailed_results_file)
            checkpoint_manager.load_checkpoint()
            postprocessor = self.deep_research_workflow.onepass_postprocessor
            if workflow == 'deep_research' and postprocessor is not None:
                committed_query_ids = set()
                for processed_idx in checkpoint_manager.processed_indices:
                    if 0 <= processed_idx < len(benchmark_data):
                        query_data = benchmark_data[processed_idx]
                        committed_query_ids.add(
                            query_data.get('qid') or query_data.get('query_id') or f'idx-{processed_idx}'
                        )
                for cached_result in checkpoint_manager.cached_results:
                    postprocess_result = cached_result.get('postprocess_results') or {}
                    query_id = postprocess_result.get('query_id') if isinstance(postprocess_result, dict) else None
                    if query_id:
                        committed_query_ids.add(query_id)
                artifact_reconciliation = postprocessor.reconcile_artifacts(
                    checkpoint_manager.processed_indices,
                    committed_query_ids,
                )
                removed_rows = artifact_reconciliation.get('removed_rows', 0)
                removed_malformed = artifact_reconciliation.get('removed_malformed_rows', 0)
                if removed_rows or removed_malformed:
                    logger.warning(
                        f"[🧹] Removed {removed_rows} uncommitted and {removed_malformed} malformed artifact rows"
                    )
                else:
                    logger.info("[✓] Artifact checkpoint reconciliation found no uncommitted rows")

        results = {
            'total_queries': len(benchmark_data),
            'successful_queries': 0,
            'prompt_type': self.prompt_type,
            'search_method': self.search_method,
            'workflow': workflow,
            'detailed_results': []
        }
        if artifact_reconciliation is not None:
            results['artifact_reconciliation'] = artifact_reconciliation
        
        # Initialize workflow-specific result containers
        if workflow == 'simple':
            for k in top_k_list:
                results[f'recall@{k}'] = []
                results[f'precision@{k}'] = []
        
        # TODO[resume]: Rebuild statistics from checkpoint if exists
        if checkpoint_manager and checkpoint_manager.cached_results:
            checkpoint_manager.rebuild_statistics(
                results=results,
                workflow=workflow,
                top_k_list=top_k_list,
                max_iterations=max_iterations
            )
        
        for idx, query_data in enumerate(tqdm(benchmark_data, desc="Evaluating queries")):
            # TODO[resume]: Skip already processed queries
            if checkpoint_manager and checkpoint_manager.is_processed(idx):
                logger.info(f"[⏭️] Skipping already processed query {idx}")
                continue
                
            try:
                if workflow == 'deep_research':
                    query_result = self.evaluate_single_query_deep_research(
                        query_data, 
                        results_per_query=results_per_query,
                        max_iterations=max_iterations,
                        idx=idx,
                    )
                    if query_result:
                        results['successful_queries'] += 1
                        results['detailed_results'].append(query_result)
                        
                        # TODO[resume]: Immediately write result to file using checkpoint manager
                        if checkpoint_manager:
                            checkpoint_manager.append_result(query_result)
                        
                        # Collect metrics by iteration (unified approach)
                        metric_names = ['recall', 'precision', 'retrieval_recall', 'retrieval_precision', 'missed_gt_ratio']
                        metrics_by_iter = {name: {} for name in metric_names}
                        
                        for res in query_result['iteration_results']:
                            it = res['iter_idx']
                            for metric_name in metric_names:
                                if metric_name in res:
                                    metrics_by_iter[metric_name][it] = res[metric_name]
                        
                        # Fill missing iterations with last value and accumulate to results
                        for metric_name, metric_dict in metrics_by_iter.items():
                            if not metric_dict:
                                continue
                            
                            max_it = max(metric_dict.keys())
                            last_val = metric_dict[max_it]
                            for it in range(max_it + 1, max_iterations + 1):
                                metric_dict[it] = last_val
                            
                            # Add to results
                            for it, val in metric_dict.items():
                                key = f'{metric_name}_iter_{it}'
                                if key not in results:
                                    results[key] = []
                                results[key].append(val)
                            
                        # 耗时统计
                        # Note: evaluation_summary.jsonl only records aggregated `avg_*` metrics.
                        # So we must accumulate all *_during fields we care about here.
                        for phase in ['planner', 'retrieval', 'selector', 'browser', 'overhead', 'total']:
                            phase_key = f'{phase}_during'
                            if phase_key not in results:
                                results[phase_key] = []
                            for res in query_result['iteration_results']:
                                time_val = res.get(phase_key, -1)
                                if time_val >= 0:
                                    results[phase_key].append(time_val)
                        
                        # 累积各轮次的 avg_distance 列表，区分轮次
                        for res in query_result['iteration_results']:
                            iter_idx = res['iter_idx']
                            distance_key = f'avg_distance_iter_{iter_idx}'
                            if distance_key not in results:
                                results[distance_key] = []
                            avg_distance = res.get('avg_distance', -1)
                            if avg_distance >= 0:
                                results[distance_key].append(avg_distance)
                        
                        # 累积各轮次的 discarded_ratio 列表，区分轮次
                        for res in query_result['iteration_results']:
                            iter_idx = res['iter_idx']
                            iteration_metrics = res.get('iteration_metrics', {})
                            
                            # Discarded ratio
                            ratio_key = f'discarded_ratio_iter_{iter_idx}'
                            if ratio_key not in results:
                                results[ratio_key] = []
                            discarded_ratio = iteration_metrics.get('discarded_ratio', -1)
                            if discarded_ratio >= 0:
                                results[ratio_key].append(discarded_ratio)
                            
                            # Discarded total count
                            count_key = f'discarded_total_count_iter_{iter_idx}'
                            if count_key not in results:
                                results[count_key] = []
                            discarded_count = iteration_metrics.get('discarded_total_count', -1)
                            if discarded_count >= 0:
                                results[count_key].append(discarded_count)

                else: # simple workflow
                    query_result = self.simple_workflow.run(
                        query_data, 
                        top_k=max(top_k_list), 
                        search_method=self.search_method,
                        idx=idx
                    )
                    if query_result:
                        results['successful_queries'] += 1
                        results['detailed_results'].append(query_result)
                        
                        # TODO[resume]: Immediately write result to file using checkpoint manager
                        if checkpoint_manager:
                            checkpoint_manager.append_result(query_result)
                        
                        gt_arxiv_ids = set(query_result['ground_truth_arxiv_ids'])
                        retrieved_arxiv_ids = [res['arxiv_id'] for res in query_result['top_results']]
                        
                        for k in top_k_list:
                            top_k_arxiv_ids = set(retrieved_arxiv_ids[:k])
                            matches_k = len(gt_arxiv_ids.intersection(top_k_arxiv_ids))
                            recall_k = matches_k / len(gt_arxiv_ids) if gt_arxiv_ids else 0.0
                            precision_k = matches_k / len(top_k_arxiv_ids) if top_k_arxiv_ids else 0.0
                            
                            results[f'recall@{k}'].append(recall_k)
                            results[f'precision@{k}'].append(precision_k)
                        
            except Exception as e:
                logger.warning(f"[💢]Failed to evaluate query {query_data.get('query', 'N/A')[:30]}: {e}")
                continue
            
        if workflow == 'deep_research':
            for key in list(results.keys()):
                if key.startswith('recall_iter_'):
                    avg_key = f'avg_{key}'
                    results[avg_key] = np.mean(results[key]) if results[key] else 0.0
                
                # 计算各轮次平均 selection precision
                elif key.startswith('precision_iter_'):
                    avg_key = f'avg_{key}'
                    results[avg_key] = np.mean(results[key]) if results[key] else 0.0
                
                elif key.startswith('retrieval_recall_iter_'):
                    avg_key = f'avg_{key}'
                    results[avg_key] = np.mean(results[key]) if results[key] else 0.0
                
                # 计算各轮次平均 retrieval precision
                elif key.startswith('retrieval_precision_iter_'):
                    avg_key = f'avg_{key}'
                    results[avg_key] = np.mean(results[key]) if results[key] else 0.0

                # 计算各阶段平均耗时
                elif key.endswith('_during'):
                    avg_key = f'avg_{key}'
                    results[avg_key] = np.mean(results[key]) if results[key] else 0.0
                
                # 计算各轮次平均 avg_distance
                elif key.startswith('avg_distance_iter_'):
                    avg_key = f'avg_{key}'
                    results[avg_key] = np.mean(results[key]) if results[key] else 0.0
                
                # 计算各轮次平均 discarded_ratio
                elif key.startswith('discarded_ratio_iter_'):
                    avg_key = f'avg_{key}'
                    results[avg_key] = np.mean(results[key]) if results[key] else 0.0
                
                # 计算各轮次平均 discarded_total_count
                elif key.startswith('discarded_total_count_iter_'):
                    avg_key = f'avg_{key}'
                    results[avg_key] = np.mean(results[key]) if results[key] else 0.0

                elif key.startswith('missed_gt_ratio_iter_'):
                    avg_key = f'avg_{key}'
                    results[avg_key] = np.mean(results[key]) if results[key] else 0.0

            # 计算各轮次平均 missed_gt_ratio 在轮次上的平均
            missed_gt_ratio_avgs = [v for k, v in results.items() if k.startswith('avg_missed_gt_ratio_iter_')]
            if missed_gt_ratio_avgs:
                results['avg_missed_gt_ratio_macro_avg'] = np.mean(missed_gt_ratio_avgs)

            results['postprocess_overall_metrics'] = aggregate_postprocess_metrics(
                results.get('detailed_results') or []
            )

        else:
            for k in top_k_list:
                results[f'avg_recall@{k}'] = np.mean(results[f'recall@{k}']) if results[f'recall@{k}'] else 0.0
                results[f'avg_precision@{k}'] = np.mean(results[f'precision@{k}']) if results[f'precision@{k}'] else 0.0
        
        return results

    def save_results(self, results: Dict, output_dir: str):
        """Save evaluation results to a JSON file with a descriptive name."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"eval_results_{results['workflow']}_{results['search_method']}_"
            f"{results['prompt_type']}_{timestamp}.json"
        )
        output_path = os.path.join(output_dir, filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        logger.info(f"[💾]Results saved to {output_path}")

    def save_detailed_results(self, detailed_results: List[Dict], output_file: str):
        """Save detailed per-query evaluation results to a JSONL file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in detailed_results:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
        logger.info(f"[📊]Detailed results saved to {output_file}")

    def append_summary_record(self, results: Dict, output_file: str, detailed_results_file: str):
        """Append a summary of the evaluation record to a file in JSONL format."""
        record = {
            "model_name": self.llm_model,
            "prompt_type": results.get('prompt_type'),
            "search_method": results.get('search_method'),
            "workflow": results.get('workflow'),
            "enable_reasoning": config.ENABLE_REASONING,
            "enable_structured_output": config.ENABLE_STRUCTURED_OUTPUT,
            "EVAL_TOP_K_VALUES": config.EVAL_TOP_K_VALUES,
            "MAX_RESULTS_PER_QUERY": config.MAX_RESULTS_PER_QUERY,
            "EVAL_MAX_ITERATIONS": config.EVAL_MAX_ITERATIONS,
            "EVAL_DETAILED_RESULTS_PATH": detailed_results_file,
            "GT_RANK_CUTOFF": config.GT_RANK_CUTOFF,
            "BROWSER_MODE": config.BROWSER_MODE,
            "PLANNER_ABLATION": config.PLANNER_ABLATION,
        }
        record.update(results.get('method_config') or {})
        
        # Flatten the avg_recalls dictionary and clean up the keys
        avg_metrics = {k: v for k, v in results.items() if k.startswith('avg_')}
        cleaned_metrics = {k.replace('avg_', ''): v for k, v in avg_metrics.items()}
        record.update(cleaned_metrics)
        if 'postprocess_overall_metrics' in results:
            record['postprocess_overall_metrics'] = results['postprocess_overall_metrics']
        
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
        
        logger.info(f"Evaluation summary record appended to {output_file}")

    def print_summary(self, results: Dict):
        """Print evaluation results summary."""
        logger.info("=" * 60)
        logger.info("📊 CITATION EVALUATION RESULTS 📊")
        logger.info("=" * 60)
        logger.info(f"Total queries: {results['total_queries']}")
        logger.info(f"Successfully evaluated: {results['successful_queries']}")
        logger.info(f"Prompt type: {results.get('prompt_type', 'N/A')}")
        logger.info(f"Search method: {results.get('search_method', 'N/A')}")
        logger.info("")
        
        for key in sorted(results.keys()):
            if key.startswith('avg_recall@'):
                k_value = key.split('@')[1]
                recall = results[key]
                precision = results.get(f'avg_precision@{k_value}', 0.0)
                logger.info(f"Average Recall@{k_value}: {recall:.4f}, Precision@{k_value}: {precision:.4f}")
        
        logger.info("=" * 60)

def main():
    """Main evaluation pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Citation RAG Evaluation System')
    parser.add_argument('--config', type=str, default=None, help='Path to config.py file (if specified, overrides default config)')
    parser.add_argument('--paper_db', type=str, default=None, help='Path to paper database JSON file')
    parser.add_argument('--benchmark_jsonl', type=str, default=None, help='Path to benchmark JSONL file')
    parser.add_argument('--llm_model', type=str, default=None, help='LLM model for query generation')
    parser.add_argument('--bm25_path', type=str, default=None, help='Path for BM25 index file')
    parser.add_argument('--output_dir', type=str, default=None, help='Base directory to save evaluation results')
    parser.add_argument('--run_label', type=str, default='', help='Optional output-directory label')
    parser.add_argument('--limit', type=int, default=None, help='Process only the first N benchmark queries')
    parser.add_argument('--rebuild_index', action='store_true', help='Rebuild the BM25 index; dense Qdrant indices are built separately')
    parser.add_argument('--top_k', type=int, nargs='+', default=None, help='Top-k values for evaluation')
    parser.add_argument('--is_local', action='store_true', default=None, help='Use local LLM API')
    parser.add_argument('--prompt_type', type=str, default=None, choices=['complex', 'simple'], help='Type of prompt to use ("complex" or "simple")')
    parser.add_argument('--search_method', type=str, default=None, choices=['vector', 'bm25'], help='Matched retrieval/rerank method')
    parser.add_argument('--workflow', type=str, default=None, choices=['simple', 'deep_research'], help='Evaluation workflow to use')
    parser.add_argument('--max_iterations', type=int, default=None, help='Maximum number of iterations for deep research workflow')
    parser.add_argument('--results_per_query', type=int, default=None, help='Results per query for deep research workflow')
    parser.add_argument('--browser_mode', type=str, default=None, choices=['PRE_ENRICH', 'REFRESH', 'INCREMENTAL', 'NONE'], help='Browser mode for deep research workflow')
    parser.add_argument('--save_level', choices=['minimal', 'full'], default='minimal')
    parser.add_argument(
        '--postprocess_stage',
        choices=['full', 'materialize'],
        default='full',
        help=(
            'full applies the selected static/dynamic rerank and shadow Selectors; '
            'materialize is Stage A and writes candidate pools/component features only'
        ),
    )
    parser.add_argument('--run_per_subquery_postprocess', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--run_deep_merged_postprocess', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--graph_method', choices=['citations', 'references', 'citations_references'], default='citations_references')
    parser.add_argument('--graph_expansion_limit', type=int, default=100)
    parser.add_argument('--graph_cache_dir', default='cache/s2_graph_oracle')
    parser.add_argument('--graph_rate_limit_rps', type=float, default=4.0)
    parser.add_argument('--graph_offline_cache_only', action='store_true')
    parser.add_argument(
        '--dynamic_rerank',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'Generate one query-conditioned policy for all OnePass graph events; '
            'disabled by default to preserve the static baseline'
        ),
    )
    parser.add_argument('--rerank_policy_model', default=None)
    parser.add_argument(
        '--rerank_policy_is_local',
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        '--rerank_policy_cache',
        default='cache/dynamic_rerank/onepass_query_policies.jsonl',
    )
    parser.add_argument('--rerank_retry_cached_fallbacks', action='store_true')
    parser.add_argument('--rerank_min_confidence', type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument(
        '--rerank_semantic_min_mass',
        type=float,
        default=DEFAULT_SEMANTIC_MIN_MASS,
    )
    parser.add_argument('--rerank_negative_weight', type=float, default=DEFAULT_NEGATIVE_WEIGHT)
    parser.add_argument(
        '--rerank_max_negative_mass',
        type=float,
        default=DEFAULT_MAX_NEGATIVE_MASS,
    )
    parser.add_argument(
        '--paper_type_cache',
        default=None,
        help='Append-only native S2 publicationTypes cache',
    )
    parser.add_argument('--paper_type_rate_limit_rps', type=float, default=1.0)
    parser.add_argument(
        '--paper_type_offline_cache_only',
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument('--embedding_base_url', default=None, help='Ollama URL for baseline dense retrieval/reranking')
    parser.add_argument('--qdrant_url', default=None)
    parser.add_argument('--qdrant_collection', default='paper_knowledge_base')
    parser.add_argument('--postprocess_event_workers', type=int, default=None, help='Bounded workers for independent graph/deep rerank events')
    parser.add_argument('--postprocess_selector_concurrency', type=int, default=None, help='Maximum concurrent shadow Selector API calls')
    parser.add_argument('--postprocess_embedding_concurrency', type=int, default=None, help='Maximum concurrent postprocess Ollama embedding calls')
    parser.add_argument('--allow_resume_compatible_code_change', action='store_true', help='Resume an audited result-preserving implementation upgrade when all recorded semantic settings match')

    args = parser.parse_args()
    if args.dynamic_rerank and args.postprocess_stage != 'full':
        parser.error('--dynamic_rerank requires --postprocess_stage full')
    
    # Load config from custom path if specified
    cfg = config
    config_path = None
    if args.config:
        config_path = args.config
        cfg = load_config_from_path(config_path)
    
    # Use command-line args if provided, otherwise fall back to loaded config (cfg)
    paper_db = args.paper_db or cfg.PAPER_DB_PATH
    benchmark_jsonl = args.benchmark_jsonl or cfg.BENCHMARK_PATH
    llm_model = args.llm_model or cfg.LLM_MODEL_NAME
    faiss_path = getattr(cfg, 'FAISS_PATH_PREFIX', '')
    bm25_path = args.bm25_path or cfg.BM25_PATH
    output_dir = args.output_dir or cfg.EVAL_BASE_DIR
    top_k = args.top_k or cfg.EVAL_TOP_K_VALUES
    is_local = args.is_local if args.is_local is not None else cfg.IS_LOCAL_LLM  # bool needs explicit None check
    prompt_type = args.prompt_type or cfg.EVAL_PROMPT_TYPE
    search_method = args.search_method or cfg.EVAL_SEARCH_METHOD
    if search_method not in {'vector', 'bm25'}:
        raise ValueError('This package supports matched bm25 or vector modes only')
    if search_method == 'vector' and args.rebuild_index:
        raise ValueError('Use code/build_vector_db_configurable.py to build a dense Qdrant index')
    workflow = args.workflow or cfg.EVAL_WORKFLOW
    max_iterations = args.max_iterations or cfg.EVAL_MAX_ITERATIONS
    results_per_query = args.results_per_query or cfg.MAX_RESULTS_PER_QUERY
    browser_mode = args.browser_mode or cfg.BROWSER_MODE
    postprocess_event_workers = (
        args.postprocess_event_workers
        if args.postprocess_event_workers is not None
        else int(getattr(cfg, 'POSTPROCESS_EVENT_WORKERS', 4))
    )
    postprocess_selector_concurrency = (
        args.postprocess_selector_concurrency
        if args.postprocess_selector_concurrency is not None
        else int(getattr(cfg, 'POSTPROCESS_SELECTOR_CONCURRENCY', 4))
    )
    postprocess_embedding_concurrency = (
        args.postprocess_embedding_concurrency
        if args.postprocess_embedding_concurrency is not None
        else int(getattr(cfg, 'POSTPROCESS_EMBEDDING_CONCURRENCY', 1))
    )
    if min(
        postprocess_event_workers,
        postprocess_selector_concurrency,
        postprocess_embedding_concurrency,
    ) < 1:
        raise ValueError('Postprocess concurrency values must all be >= 1')

    # The workflow modules import the canonical config module, so mirror the
    # resolved custom-config/CLI values before constructing agents.
    config.LLM_MODEL_NAME = llm_model
    config.IS_LOCAL_LLM = is_local
    config.LLM_GEN_PARAMS = cfg.LLM_GEN_PARAMS
    config.ENABLE_REASONING = cfg.ENABLE_REASONING
    config.ENABLE_STRUCTURED_OUTPUT = cfg.ENABLE_STRUCTURED_OUTPUT
    config.MAX_RESULTS_PER_QUERY = results_per_query
    config.EVAL_MAX_ITERATIONS = max_iterations
    config.BROWSER_MODE = browser_mode
    config.PLANNER_ABLATION = getattr(cfg, 'PLANNER_ABLATION', False)
    config.SAVE_AGENT_TRACES = False
    config.DEBUG = False
    any_shadow = (
        args.run_per_subquery_postprocess
        or args.run_deep_merged_postprocess
    )
    if args.run_deep_merged_postprocess and not args.run_per_subquery_postprocess:
        raise ValueError('Deep shadows require --run_per_subquery_postprocess because graph local-pool sizes define their budgets')
    if args.dynamic_rerank and not args.run_per_subquery_postprocess:
        raise ValueError('--dynamic_rerank requires --run_per_subquery_postprocess')
    if any_shadow and browser_mode != 'NONE':
        raise ValueError('One-pass replay postprocessors currently require --browser_mode NONE')
    if any_shadow and getattr(cfg, 'ENABLE_SUMMARIZATION', False):
        raise ValueError('One-pass replay requires ENABLE_SUMMARIZATION=False so only the candidate list changes')

    # Use loaded config (cfg) for flags, not global config
    reasoning_flag = 'reasoning' if cfg.ENABLE_REASONING else 'instruct'
    structured_flag = 'structured' if cfg.ENABLE_STRUCTURED_OUTPUT else 'non-structured'
    ablation_flag = '_ablation' if getattr(cfg, 'PLANNER_ABLATION', False) else ''
    model_name = llm_model.split('/')[-1] if '/' in llm_model else llm_model
    label_suffix = f"_{args.run_label}" if args.run_label else ''
    stage_suffix = '_stage-materialize' if args.postprocess_stage == 'materialize' else ''
    rerank_suffix = (
        '_dynamic-rerank-v1_type-s2-native'
        if args.dynamic_rerank
        else '_static-rerank'
    )
    current_output_dir = os.path.join(output_dir, f"{model_name}_{prompt_type}_{search_method}_{workflow}_topk-{top_k}_maxq-{results_per_query}_{reasoning_flag}_{structured_flag}_{browser_mode}{ablation_flag}{stage_suffix}{rerank_suffix}{label_suffix}")
    os.makedirs(current_output_dir, exist_ok=True)

    source_config = args.config if args.config else config.__file__
    canonical_embedding_model = getattr(
        config, 'OLLAMA_EMBEDDING_MODEL', 'qwen3-embedding:0.6b'
    )
    configured_embedding_model = getattr(
        cfg, 'OLLAMA_EMBEDDING_MODEL', canonical_embedding_model
    )
    if search_method == 'vector' and configured_embedding_model != canonical_embedding_model:
        raise ValueError(
            f"Dense mode is fixed to baseline model {canonical_embedding_model!r}; "
            f"custom config requested {configured_embedding_model!r}"
        )
    embedding_model_name = canonical_embedding_model if search_method == 'vector' else None
    embedding_base_url = (
        args.embedding_base_url or getattr(cfg, 'OLLAMA_URL', 'http://localhost:11434')
        if search_method == 'vector'
        else None
    )
    qdrant_url = (
        args.qdrant_url or getattr(cfg, 'QDRANT_URL', None)
        if search_method == 'vector'
        else None
    )
    rerank_policy_model = (
        args.rerank_policy_model
        or os.environ.get('SCHOLARGYM_MODEL')
        or llm_model
    )
    rerank_policy_is_local = (
        args.rerank_policy_is_local
        if args.rerank_policy_is_local is not None
        else is_local
    )
    paper_type_offline = (
        args.paper_type_offline_cache_only
        if args.paper_type_offline_cache_only is not None
        else args.graph_offline_cache_only
    )
    paper_type_cache = args.paper_type_cache or (
        'cache/dynamic_rerank/onepass_paper_types_s2_native.jsonl'
    )
    package_source_files = [
        'api.py', 'config.py', 'deeprag.py', 'dimension_catalog.py', 'eval.py',
        'graph_methods.py', 'metrics.py', 'deep_retrieval.py',
        'online_paper_type.py', 'onepass_postprocess.py', 'paper_type.py',
        'prompt.py', 'rag.py', 'rerank_skill.py', 'simplerag.py',
        'structures.py', 'utils.py',
        os.path.join('agent', 'browser.py'),
        os.path.join('agent', 'planner.py'),
        os.path.join('agent', 'selector.py'),
        os.path.join('agent', 'summarizer.py'),
        os.path.join('mcp', 'retrieval_mcp.py'),
    ]
    package_hash = package_source_sha256(package_source_files)
    run_signature_payload = {
        'upstream_commit': 'f426fd15e3ff28ee11ddeafc253dffd73ef88500',
        'artifact_schema_version': '1.4',
        'package_source_sha256': package_hash,
        'config': file_identity(source_config, include_sha256=True),
        'benchmark': file_identity(benchmark_jsonl, include_sha256=True),
        'paper_db': file_identity(paper_db),
        'bm25_index': (
            {'path': os.path.realpath(bm25_path), 'rebuild_each_start': True}
            if search_method == 'bm25' and args.rebuild_index
            else (file_identity(bm25_path) if search_method == 'bm25' else None)
        ),
        'llm_model': llm_model,
        'is_local_llm': is_local,
        'llm_gen_params': cfg.LLM_GEN_PARAMS,
        'prompt_type': prompt_type,
        'workflow': workflow,
        'top_k': top_k,
        'max_iterations': max_iterations,
        'results_per_query': results_per_query,
        'browser_mode': browser_mode,
        'planner_ablation': getattr(cfg, 'PLANNER_ABLATION', False),
        'enable_reasoning': cfg.ENABLE_REASONING,
        'enable_structured_output': cfg.ENABLE_STRUCTURED_OUTPUT,
        'search_method': search_method,
        'embedding_model': embedding_model_name,
        'paper_embedding_serialization_id': (
            PAPER_EMBEDDING_SERIALIZATION_ID
            if search_method == 'vector'
            else None
        ),
        'embedding_base_url': embedding_base_url,
        'qdrant_url': qdrant_url,
        'qdrant_collection': args.qdrant_collection if search_method == 'vector' else None,
        'save_level': args.save_level,
        'postprocess_stage': args.postprocess_stage,
        'zero_ground_truth_policy': (
            'materialize_pools_but_exclude_from_pool_recall_aggregates'
            if args.postprocess_stage == 'materialize'
            else 'skip_query_before_workflow'
        ),
        'run_per_subquery_postprocess': args.run_per_subquery_postprocess,
        'run_deep_merged_postprocess': args.run_deep_merged_postprocess,
        'graph_method': args.graph_method,
        'graph_expansion_limit': args.graph_expansion_limit,
        'graph_rate_limit_rps': args.graph_rate_limit_rps,
        'graph_cache_dir': os.path.realpath(args.graph_cache_dir),
        'graph_offline_cache_only': args.graph_offline_cache_only,
        'dynamic_rerank_requested': args.dynamic_rerank,
        'rerank_policy_model': rerank_policy_model if args.dynamic_rerank else None,
        'rerank_policy_is_local': rerank_policy_is_local if args.dynamic_rerank else None,
        'rerank_policy_cache': (
            os.path.realpath(args.rerank_policy_cache)
            if args.dynamic_rerank
            else None
        ),
        'rerank_retry_cached_fallbacks': (
            args.rerank_retry_cached_fallbacks if args.dynamic_rerank else None
        ),
        'rerank_min_confidence': (
            args.rerank_min_confidence if args.dynamic_rerank else None
        ),
        'rerank_semantic_min_mass': (
            args.rerank_semantic_min_mass if args.dynamic_rerank else None
        ),
        'rerank_negative_weight': (
            args.rerank_negative_weight if args.dynamic_rerank else None
        ),
        'rerank_max_negative_mass': (
            args.rerank_max_negative_mass if args.dynamic_rerank else None
        ),
        'paper_type_backend': (
            PAPER_TYPE_BACKEND if args.dynamic_rerank else None
        ),
        'paper_type_namespace': (
            S2_NATIVE_PAPER_TYPE_NAMESPACE if args.dynamic_rerank else None
        ),
        'paper_type_cache': (
            os.path.realpath(paper_type_cache) if args.dynamic_rerank else None
        ),
        'paper_type_offline_cache_only': (
            paper_type_offline if args.dynamic_rerank else None
        ),
        'paper_type_rate_limit_rps': (
            args.paper_type_rate_limit_rps
            if args.dynamic_rerank
            else None
        ),
        'rerank_catalog_version': CATALOG_VERSION if args.dynamic_rerank else None,
        'rerank_prompt_version': PROMPT_VERSION if args.dynamic_rerank else None,
        'feature_weights_applied': (
            dict(DEFAULT_FEATURE_WEIGHTS)
            if args.postprocess_stage == 'full' and not args.dynamic_rerank
            else None
        ),
        'rerank_formula_id': (
            (
                POLICY_VERSION
                if args.dynamic_rerank
                else RERANK_FORMULA_ID
            )
            if args.postprocess_stage == 'full'
            else None
        ),
        'materialized_feature_names': {
            'graph': [
                'query_score_raw', 'query_score_normalized', 'query_component_rank',
                'subquery_score_raw', 'subquery_score_normalized', 'subquery_component_rank',
                'intent_labels', 'intent_score', 'path_count', 'path_count_normalized',
            ],
            'deep': [
                'deep_retrieval_score_raw', 'deep_retrieval_rank_global_date_valid',
                'deep_retrieval_rank_after_exclusion', 'deep_retrieval_rank_in_local_pool',
                'query_score_raw', 'query_score_normalized', 'query_component_rank',
                'subquery_score_raw', 'subquery_score_normalized', 'subquery_component_rank',
            ],
        },
        'intent_weights': dict(INTENT_WEIGHTS),
        'local_rerank_embedding_batch_size': getattr(config, 'LOCAL_RERANK_EMBEDDING_BATCH_SIZE', 64),
        'deep_vector_max_fetch_k': getattr(config, 'DEEP_VECTOR_MAX_FETCH_K', 20000),
        'postprocess_event_workers': postprocess_event_workers,
        'postprocess_selector_concurrency': postprocess_selector_concurrency,
        'postprocess_embedding_concurrency': postprocess_embedding_concurrency,
        'postprocess_embedding_cache_policy': (
            QUERY_SCOPED_EMBEDDING_CACHE_POLICY
            if search_method == 'vector'
            else None
        ),
        'limit': args.limit,
    }
    run_signature = signature_sha256(run_signature_payload)
    detailed_results_file = os.path.join(current_output_dir, 'detailed_results.jsonl')
    manifest_path = os.path.join(current_output_dir, 'onepass_artifacts', 'run_manifest.json')
    resume_validation = validate_resume_signature(
        manifest_path,
        detailed_results_file,
        run_signature,
        expected_payload=run_signature_payload,
        allow_compatible_code_change=args.allow_resume_compatible_code_change,
    )
    if resume_validation.get("mode") == "compatible_code_change":
        logger.warning(
            "[⚠️] Resuming after an explicitly allowed compatible package-code "
            "upgrade; all recorded data/model/method settings matched"
        )

    # Save config file for reproduction
    try:
        if source_config:
            target_config_path = os.path.join(current_output_dir, "config.py")
            shutil.copy(source_config, target_config_path)
            logger.info(f"[💾] Config file saved to {target_config_path}")
    except Exception as e:
        logger.warning(f"[⚠️] Failed to save config file: {e}")

    config.CASE_STUDY_OUTPUT_DIR = os.path.join(current_output_dir, "case_study")
    # TODO: different workflow should have different output dir, simple workflow use top_k instead of results_per_query

    logger.info("[🚀]Initializing RAG system...")
    embedding_provider = None
    if search_method == 'vector':
        embedding_provider = OllamaEmbeddings(
            model=embedding_model_name,
            base_url=embedding_base_url,
        )
    postprocess_embedding_provider = (
        QueryScopedEmbeddingCache(
            BoundedEmbeddingProvider(
                embedding_provider,
                max_concurrency=postprocess_embedding_concurrency,
            )
        )
        if embedding_provider is not None
        else None
    )
    rag_system = CitationRAGSystem(
        search_method=search_method,
        embedding_provider=embedding_provider,
        qdrant_url=qdrant_url,
        qdrant_collection=args.qdrant_collection,
    )
    
    rag_system.load_or_build_indices(
        paper_db_path=paper_db,
        faiss_path=faiss_path,
        bm25_path=bm25_path,
        rebuild=args.rebuild_index
    )
    
    # Initialize trace recorder if enabled
    trace_recorder = None
    if config.SAVE_AGENT_TRACES:
        trace_recorder = AgentTraceRecorder(
            output_dir=output_dir,
            model_name=llm_model,
            prompt_type=prompt_type,
            search_method=search_method,
            workflow=workflow,
            top_k=top_k,
            max_results=results_per_query,
            enable_reasoning=cfg.ENABLE_REASONING,
            enable_structured=cfg.ENABLE_STRUCTURED_OUTPUT
        )
    
    logger.info("[🚀]Initializing evaluator...")
    artifacts_dir = os.path.join(current_output_dir, 'onepass_artifacts')
    artifact_writer = ArtifactWriter(artifacts_dir, args.save_level)
    paper_db_index = load_paper_db(paper_db)
    s2_client = S2GraphClient(
        args.graph_cache_dir,
        rate_limit_rps=args.graph_rate_limit_rps,
        offline=args.graph_offline_cache_only,
    )
    scoring_backend = 'embedding' if search_method == 'vector' else 'bm25'
    rerank_skill = None
    paper_type_resolver = None
    if args.dynamic_rerank:
        paper_type_resolver = S2PublicationTypeResolver(
            paper_type_cache,
            requests_per_second=args.paper_type_rate_limit_rps,
            offline=paper_type_offline,
        )
        rerank_skill = RerankSkill(
            rerank_policy_model,
            is_local=rerank_policy_is_local,
            policy_cache_path=args.rerank_policy_cache,
            retry_cached_fallbacks=args.rerank_retry_cached_fallbacks,
            paper_type_cache=paper_type_resolver.snapshot_cache(),
            min_confidence=args.rerank_min_confidence,
            semantic_min_mass=args.rerank_semantic_min_mass,
            negative_weight=args.rerank_negative_weight,
            max_negative_mass=args.rerank_max_negative_mass,
        )
    per_subquery_processor = PerSubqueryProcessor(
        paper_db_index,
        s2_client,
        scoring_backend=scoring_backend,
        embedding_provider=postprocess_embedding_provider,
        expansion_method=args.graph_method,
        expansion_limit=args.graph_expansion_limit,
        rerank_skill=rerank_skill,
        paper_type_resolver=paper_type_resolver,
    )
    deep_retrieval_processor = DeepRetrievalProcessor(
        rag_system,
        paper_db_index,
        scoring_backend=scoring_backend,
        embedding_provider=postprocess_embedding_provider,
    )
    onepass_postprocessor = OnePassPostprocessor(
        selector=None,
        paper_db=paper_db_index,
        writer=artifact_writer,
        s2_client=s2_client,
        per_subquery_processor=per_subquery_processor,
        deep_retrieval_processor=deep_retrieval_processor,
        scoring_backend=scoring_backend,
        embedding_provider=postprocess_embedding_provider,
        run_per_subquery=args.run_per_subquery_postprocess,
        run_deep_merged=args.run_deep_merged_postprocess,
        postprocess_stage=args.postprocess_stage,
        run_id=os.path.basename(current_output_dir),
        event_workers=postprocess_event_workers,
        selector_concurrency=postprocess_selector_concurrency,
    )
    artifact_writer.write_json('run_manifest.json', {
        'upstream_repository': 'https://github.com/shenhao-stu/ScholarGym.git',
        'baseline_commit_mirror': 'https://github.com/qyx687/ScholarGym.git@baseline-repro',
        'upstream_commit': 'f426fd15e3ff28ee11ddeafc253dffd73ef88500',
        'artifact_schema_version': '1.4',
        'artifact_write_mode': 'query_staging_then_flat_jsonl_commit',
        'artifact_checkpoint_source': 'detailed_results.jsonl',
        'package_source_sha256': package_hash,
        'run_signature_sha256': run_signature,
        'run_signature': run_signature_payload,
        'resume_validation': resume_validation,
        'save_level': args.save_level,
        'postprocess_stage': args.postprocess_stage,
        'stage_a_only': args.postprocess_stage == 'materialize',
        'stage_a_query_commit_policy': (
            'all_enabled_materializers_must_complete'
            if args.postprocess_stage == 'materialize'
            else None
        ),
        'zero_ground_truth_policy': (
            'materialize_pools_but_exclude_from_pool_recall_aggregates'
            if args.postprocess_stage == 'materialize'
            else 'skip_query_before_workflow'
        ),
        'legacy_rerank_applied': (
            args.postprocess_stage == 'full' and not args.dynamic_rerank
        ),
        'dynamic_rerank_requested': args.dynamic_rerank,
        'rerank_formula_id': (
            (
                POLICY_VERSION if args.dynamic_rerank else RERANK_FORMULA_ID
            )
            if args.postprocess_stage == 'full'
            else None
        ),
        'shadow_selector_applied': args.postprocess_stage == 'full',
        'config_path': args.config,
        'paper_db_path': paper_db,
        'benchmark_jsonl_path': benchmark_jsonl,
        'bm25_path': bm25_path if search_method == 'bm25' else None,
        'llm_model': llm_model,
        'prompt_type': prompt_type,
        'max_iterations': max_iterations,
        'browser_mode': browser_mode,
        'search_method': search_method,
        'scoring_backend': scoring_backend,
        'embedding_backend': 'ollama' if embedding_provider else None,
        'embedding_implementation': 'langchain_ollama.OllamaEmbeddings' if embedding_provider else None,
        'embedding_model': embedding_model_name,
        'paper_embedding_serialization_id': (
            PAPER_EMBEDDING_SERIALIZATION_ID if embedding_provider else None
        ),
        'embedding_base_url': embedding_base_url,
        'local_rerank_embedding_batch_size': getattr(config, 'LOCAL_RERANK_EMBEDDING_BATCH_SIZE', 64),
        'postprocess_event_workers': postprocess_event_workers,
        'postprocess_selector_concurrency': postprocess_selector_concurrency,
        'postprocess_embedding_concurrency': postprocess_embedding_concurrency,
        'postprocess_embedding_cache_policy': (
            QUERY_SCOPED_EMBEDDING_CACHE_POLICY
            if embedding_provider
            else None
        ),
        'deep_vector_max_fetch_k': getattr(config, 'DEEP_VECTOR_MAX_FETCH_K', 20000),
        'qdrant_url': qdrant_url,
        'qdrant_collection': args.qdrant_collection,
        'graph_method': args.graph_method,
        'graph_cache_dir': args.graph_cache_dir,
        'graph_expansion_limit': args.graph_expansion_limit,
        'graph_rate_limit_rps': args.graph_rate_limit_rps,
        'graph_offline_cache_only': args.graph_offline_cache_only,
        'rerank_policy_model': rerank_policy_model if args.dynamic_rerank else None,
        'rerank_policy_is_local': rerank_policy_is_local if args.dynamic_rerank else None,
        'rerank_policy_cache': (
            args.rerank_policy_cache if args.dynamic_rerank else None
        ),
        'rerank_min_confidence': (
            args.rerank_min_confidence if args.dynamic_rerank else None
        ),
        'rerank_semantic_min_mass': (
            args.rerank_semantic_min_mass if args.dynamic_rerank else None
        ),
        'rerank_negative_weight': (
            args.rerank_negative_weight if args.dynamic_rerank else None
        ),
        'rerank_max_negative_mass': (
            args.rerank_max_negative_mass if args.dynamic_rerank else None
        ),
        'paper_type_backend': (
            PAPER_TYPE_BACKEND if args.dynamic_rerank else None
        ),
        'paper_type_namespace': (
            S2_NATIVE_PAPER_TYPE_NAMESPACE if args.dynamic_rerank else None
        ),
        'paper_type_evidence_source': getattr(
            paper_type_resolver, 'evidence_source', None
        ),
        'paper_type_classifier_version': getattr(
            paper_type_resolver, 'classifier_version', None
        ),
        'paper_type_supported_types': list(
            sorted(getattr(rerank_skill, 'paper_type_supported_types', ()) or ())
        ),
        'paper_type_cache': paper_type_cache if args.dynamic_rerank else None,
        'paper_type_offline_cache_only': (
            paper_type_offline if args.dynamic_rerank else None
        ),
        'rerank_catalog_version': CATALOG_VERSION if args.dynamic_rerank else None,
        'rerank_prompt_version': PROMPT_VERSION if args.dynamic_rerank else None,
        'date_policy': 'seeds_trust_retriever_expanded_require_db_date_lte_cutoff',
        'deep_date_policy': (
            'retrieved_papers_require_nonmissing_date_lte_subquery_cutoff'
            if args.run_deep_merged_postprocess
            else None
        ),
        'run_per_subquery_postprocess': args.run_per_subquery_postprocess,
        'run_deep_merged_postprocess': args.run_deep_merged_postprocess,
        'deep_merged_budget': (
            'sum of matching graph local-pool sizes for stable subquery_id'
            if args.run_deep_merged_postprocess
            else None
        ),
        'deep_rerank_graph_features': (
            {'intent_score': 0.0, 'path_count_normalized': 0.0}
            if (
                args.postprocess_stage == 'full'
                and args.run_deep_merged_postprocess
            )
            else None
        ),
        'results_per_query': results_per_query,
        'run_label': args.run_label,
        'limit': args.limit,
        'feature_weights_applied': (
            per_subquery_processor.weights
            if args.postprocess_stage == 'full' and not args.dynamic_rerank
            else None
        ),
        'materialized_feature_names': run_signature_payload['materialized_feature_names'],
        'intent_weights': dict(INTENT_WEIGHTS),
        'prompts_saved': False,
        'paper_identity_in_artifacts': 'arxiv_id_only',
    })
    evaluator = CitationEvaluator(
        rag_system=rag_system,
        llm_model=llm_model,
        is_local=is_local,
        prompt_type=prompt_type,
        search_method=search_method,
        trace_recorder=trace_recorder,
        onepass_postprocessor=onepass_postprocessor,
    )
    
    # Load and process benchmark data
    benchmark_data = evaluator.load_benchmark_data(benchmark_jsonl)
    if args.limit is not None:
        benchmark_data = benchmark_data[:max(0, args.limit)]
    
    summary_file = os.path.join(output_dir, 'evaluation_summary.jsonl')
    
    logger.info("[📈]Starting evaluation...")
    results = evaluator.evaluate_benchmark(
        benchmark_data=benchmark_data,
        workflow=workflow,
        top_k_list=top_k,
        results_per_query=results_per_query,
        max_iterations=max_iterations,
        detailed_results_file=detailed_results_file,
        enable_resume=True
    )
    results['method_config'] = {
        'PACKAGE_METHOD': (
            'onepass_stage_a_pool_feature_materialization'
            if args.postprocess_stage == 'materialize'
            else (
                (
                    'onepass_query_conditioned_graph_rerank_with_deep_merged_control'
                    if args.run_deep_merged_postprocess
                    else 'onepass_query_conditioned_graph_rerank'
                )
                if args.dynamic_rerank
                else (
                    'onepass_static_graph_rerank_with_deep_merged_control'
                    if args.run_deep_merged_postprocess
                    else 'onepass_static_graph_rerank'
                )
            )
        ),
        'POSTPROCESS_STAGE': args.postprocess_stage,
        'ARTIFACT_WRITE_MODE': 'query_staging_then_flat_jsonl_commit',
        'SAVE_LEVEL': args.save_level,
        'RUN_PER_SUBQUERY_POSTPROCESS': args.run_per_subquery_postprocess,
        'RUN_DEEP_MERGED_POSTPROCESS': args.run_deep_merged_postprocess,
        'DYNAMIC_RERANK': args.dynamic_rerank,
        'PAPER_TYPE_BACKEND': (
            PAPER_TYPE_BACKEND if args.dynamic_rerank else None
        ),
        'PAPER_TYPE_NAMESPACE': (
            S2_NATIVE_PAPER_TYPE_NAMESPACE if args.dynamic_rerank else None
        ),
        'PAPER_TYPE_CACHE': paper_type_cache if args.dynamic_rerank else None,
        'RERANK_POLICY_CACHE': (
            args.rerank_policy_cache if args.dynamic_rerank else None
        ),
        'GRAPH_METHOD': args.graph_method,
        'GRAPH_EXPANSION_LIMIT': args.graph_expansion_limit,
        'GRAPH_RATE_LIMIT_RPS': args.graph_rate_limit_rps,
        'RERANK_FORMULA_ID': (
            (
                POLICY_VERSION if args.dynamic_rerank else RERANK_FORMULA_ID
            )
            if args.postprocess_stage == 'full'
            else None
        ),
        'DEEP_RERANK_FEATURE_WEIGHTS': (
            deep_retrieval_processor.weights
            if (
                args.postprocess_stage == 'full'
                and args.run_deep_merged_postprocess
            )
            else None
        ),
        'RERANK_FEATURE_WEIGHTS': (
            per_subquery_processor.weights
            if args.postprocess_stage == 'full' and not args.dynamic_rerank
            else None
        ),
        'MATERIALIZED_FEATURE_NAMES': run_signature_payload['materialized_feature_names'],
        'EMBEDDING_BACKEND': 'ollama' if embedding_provider else None,
        'EMBEDDING_MODEL': embedding_model_name,
        'PAPER_EMBEDDING_SERIALIZATION_ID': (
            PAPER_EMBEDDING_SERIALIZATION_ID if embedding_provider else None
        ),
        'LOCAL_RERANK_EMBEDDING_BATCH_SIZE': getattr(config, 'LOCAL_RERANK_EMBEDDING_BATCH_SIZE', 64),
        'DEEP_VECTOR_MAX_FETCH_K': getattr(config, 'DEEP_VECTOR_MAX_FETCH_K', 20000),
        'POSTPROCESS_EVENT_WORKERS': postprocess_event_workers,
        'POSTPROCESS_SELECTOR_CONCURRENCY': postprocess_selector_concurrency,
        'POSTPROCESS_EMBEDDING_CONCURRENCY': postprocess_embedding_concurrency,
        'POSTPROCESS_EMBEDDING_CACHE_POLICY': (
            QUERY_SCOPED_EMBEDDING_CACHE_POLICY
            if embedding_provider
            else None
        ),
        'QDRANT_COLLECTION': args.qdrant_collection if embedding_provider else None,
        'RUN_LABEL': args.run_label,
    }

    evaluator.print_summary(results)
    
    evaluator.save_results(results, current_output_dir)

    evaluator.append_summary_record(results, summary_file, detailed_results_file)

    logger.info(f"[✅] Evaluation completed! Results in: {current_output_dir}")

    logger.info(f"[📊] Summary record saved to: {summary_file}")


if __name__ == "__main__":
    main()
