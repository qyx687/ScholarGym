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

from logger import get_logger
from rag import CitationRAGSystem
import config
from deeprag import DeepResearchWorkflow, RANK_METRIC_SCHEMA_VERSION
from simplerag import SimpleWorkflow
from utils import extract_ground_truth_arxiv_ids, CheckpointManager, calculate_retrieval_metrics, AgentTraceRecorder
from graph_methods import ArtifactWriter, EmbeddingProvider, PerSubqueryProcessor, S2GraphClient, load_paper_db
from online_per_subquery import OnlinePerSubqueryManager

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
                 search_method: str = config.EVAL_SEARCH_METHOD, trace_recorder=None, online_per_subquery_manager=None):
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
            online_per_subquery_manager=online_per_subquery_manager,
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

        if not gt_arxiv_ids:
            return None

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
        select_steps = [item for item in workflow_results['history'] if item.get('stage') == 'select']

        # Track cumulative retrieval and selection across iterations
        all_retrieved_arxiv_ids = set()
        all_selected_arxiv_ids = set()
        total_discarded_gt_count = 0  # Cumulative count of discarded ground truth papers

        for step in select_steps:
            iter_idx = step.get('iter_idx')
            retrieved_in_iter = step.get('retrieved_papers') or {}
            selected_in_iter = step.get('selected_papers') or {}
            gt_rank = step.get('gt_rank') or []
            local_gt_rank = step.get('local_gt_rank') or []
            browsing_arxiv_ids = step.get('browsing_arxiv_ids') or []
            avg_distance = step.get('avg_distance', -1)
            local_avg_distance = step.get('local_avg_distance', -1)
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
                "gt_rank_scope": step.get('gt_rank_scope', 'baseline_retriever_rank_after_date_and_selection_exclusion'),
                "local_gt_rank": local_gt_rank,
                'local_avg_distance': local_avg_distance,
                "local_gt_rank_scope": step.get('local_gt_rank_scope', 'seed_plus_expanded_closed_pool_rerank'),
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
            'rank_metric_schema_version': workflow_results.get(
                'rank_metric_schema_version', RANK_METRIC_SCHEMA_VERSION
            ),
            'query': query,
            'ground_truth_arxiv_ids': list(gt_arxiv_ids),
            'iteration_results': iteration_results,
            'final_report': workflow_results.get('final_report', ''),
            'final_selected_papers': final_selected_papers,
            'executed_queries': workflow_results.get('executed_queries', []),
            'online_per_subquery_summary': workflow_results.get('online_per_subquery_summary', {}),
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
        if enable_resume and detailed_results_file:
            checkpoint_manager = CheckpointManager(detailed_results_file)
            checkpoint_manager.load_checkpoint()
            graph_enabled = self.deep_research_workflow.online_per_subquery_manager is not None
            incompatible_indices = [
                result.get('idx')
                for result in checkpoint_manager.cached_results
                if workflow == 'deep_research'
                and graph_enabled
                and result.get('rank_metric_schema_version') != RANK_METRIC_SCHEMA_VERSION
            ]
            if incompatible_indices:
                raise ValueError(
                    'The existing detailed_results.jsonl predates separate global/local rank metrics '
                    f'(incompatible query indices: {incompatible_indices[:10]}). Use a new --run_label '
                    'or output directory; the global retriever ranks cannot be reconstructed from the old checkpoint.'
                )

        results = {
            'total_queries': len(benchmark_data),
            'successful_queries': 0,
            'prompt_type': self.prompt_type,
            'search_method': self.search_method,
            'workflow': workflow,
            'detailed_results': []
        }
        
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
                        
                        # Accumulate independent global-retriever and local-rerank distances.
                        for res in query_result['iteration_results']:
                            iter_idx = res['iter_idx']
                            for field_name in ('avg_distance', 'local_avg_distance'):
                                distance = res.get(field_name, -1)
                                if distance >= 0:
                                    distance_key = f'{field_name}_iter_{iter_idx}'
                                    if distance_key not in results:
                                        results[distance_key] = []
                                    results[distance_key].append(distance)
                        
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

                elif key.startswith('local_avg_distance_iter_'):
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
        cleaned_metrics = {k.replace('avg_', '', 1): v for k, v in avg_metrics.items()}
        record.update(cleaned_metrics)
        
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
    parser.add_argument('--embedding_model', type=str, default=None, help='Path to embedding model')
    parser.add_argument('--llm_model', type=str, default=None, help='LLM model for query generation')
    parser.add_argument('--faiss_path', type=str, default=None, help='Path prefix for FAISS index files')
    parser.add_argument('--bm25_path', type=str, default=None, help='Path for BM25 index file')
    parser.add_argument('--output_dir', type=str, default=None, help='Base directory to save evaluation results')
    parser.add_argument('--run_label', type=str, default='', help='Optional output-directory label')
    parser.add_argument('--limit', type=int, default=None, help='Process only the first N benchmark queries')
    parser.add_argument('--rebuild_index', action='store_true', help='Force rebuild of indices')
    parser.add_argument('--top_k', type=int, nargs='+', default=None, help='Top-k values for evaluation')
    parser.add_argument('--device', type=str, default=None, help='Device for embedding model')
    parser.add_argument('--is_local', action='store_true', default=None, help='Use local LLM API')
    parser.add_argument('--prompt_type', type=str, default=None, choices=['complex', 'simple'], help='Type of prompt to use ("complex" or "simple")')
    parser.add_argument('--search_method', type=str, default=None, choices=['vector', 'bm25', 'hybrid'], help='Search method to use ("vector", "bm25", or "hybrid")')
    parser.add_argument('--workflow', type=str, default=None, choices=['simple', 'deep_research'], help='Evaluation workflow to use')
    parser.add_argument('--max_iterations', type=int, default=None, help='Maximum number of iterations for deep research workflow')
    parser.add_argument('--results_per_query', type=int, default=None, help='Results per query for deep research workflow')
    parser.add_argument('--browser_mode', type=str, default=None, choices=['PRE_ENRICH', 'REFRESH', 'INCREMENTAL', 'NONE'], help='Browser mode for deep research workflow')
    parser.add_argument('--enable_per_subquery_graph', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--save_level', choices=['minimal', 'full'], default='minimal')
    parser.add_argument('--graph_method', choices=['citations', 'references', 'citations_references'], default='citations_references')
    parser.add_argument('--graph_expansion_limit', type=int, default=100)
    parser.add_argument('--graph_cache_dir', default='cache/s2_graph_oracle')
    parser.add_argument('--graph_rate_limit_rps', type=float, default=4.0)
    parser.add_argument('--graph_offline_cache_only', action='store_true')
    parser.add_argument('--embedding_backend', choices=['ollama', 'api'], default='ollama')
    parser.add_argument('--embedding_service_model', default='qwen3-embedding:0.6b')
    parser.add_argument('--embedding_base_url', default=None)
    parser.add_argument('--embedding_api_key_env', default='EMBEDDING_API_KEY')
    parser.add_argument('--embedding_batch_size', type=int, default=64)
    parser.add_argument('--qdrant_url', default=None)
    parser.add_argument('--qdrant_collection', default='paper_knowledge_base')

    args = parser.parse_args()
    
    # Load config from custom path if specified
    cfg = config
    config_path = None
    if args.config:
        config_path = args.config
        cfg = load_config_from_path(config_path)
    
    # Use command-line args if provided, otherwise fall back to loaded config (cfg)
    paper_db = args.paper_db or cfg.PAPER_DB_PATH
    benchmark_jsonl = args.benchmark_jsonl or cfg.BENCHMARK_PATH
    embedding_model = args.embedding_model or cfg.EMBEDDING_MODEL_PATH
    llm_model = args.llm_model or cfg.LLM_MODEL_NAME
    faiss_path = args.faiss_path or cfg.FAISS_PATH_PREFIX
    bm25_path = args.bm25_path or cfg.BM25_PATH
    output_dir = args.output_dir or cfg.EVAL_BASE_DIR
    top_k = args.top_k or cfg.EVAL_TOP_K_VALUES
    device = args.device or cfg.DEVICE
    is_local = args.is_local if args.is_local is not None else cfg.IS_LOCAL_LLM  # bool needs explicit None check
    prompt_type = args.prompt_type or cfg.EVAL_PROMPT_TYPE
    search_method = args.search_method or cfg.EVAL_SEARCH_METHOD
    workflow = args.workflow or cfg.EVAL_WORKFLOW
    max_iterations = args.max_iterations or cfg.EVAL_MAX_ITERATIONS
    results_per_query = args.results_per_query or cfg.MAX_RESULTS_PER_QUERY
    browser_mode = args.browser_mode or cfg.BROWSER_MODE

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

    # Use loaded config (cfg) for flags, not global config
    reasoning_flag = 'reasoning' if cfg.ENABLE_REASONING else 'instruct'
    structured_flag = 'structured' if cfg.ENABLE_STRUCTURED_OUTPUT else 'non-structured'
    ablation_flag = '_ablation' if getattr(cfg, 'PLANNER_ABLATION', False) else ''
    model_name = llm_model.split('/')[-1] if '/' in llm_model else llm_model
    method_suffix = '_per_subquery_online' if args.enable_per_subquery_graph else '_baseline'
    label_suffix = f"_{args.run_label}" if args.run_label else ''
    current_output_dir = os.path.join(output_dir, f"{model_name}_{prompt_type}_{search_method}_{workflow}_topk-{top_k}_maxq-{results_per_query}_{reasoning_flag}_{structured_flag}_{browser_mode}{ablation_flag}{method_suffix}{label_suffix}")
    os.makedirs(current_output_dir, exist_ok=True)

    # Save config file for reproduction
    try:
        source_config = args.config if args.config else config.__file__
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
    if search_method in {'vector', 'hybrid'}:
        embedding_base_url = args.embedding_base_url or (
            getattr(cfg, 'OLLAMA_URL', 'http://localhost:11434')
            if args.embedding_backend == 'ollama'
            else 'https://openrouter.ai/api/v1'
        )
        embedding_provider = EmbeddingProvider(
            backend=args.embedding_backend,
            model=args.embedding_service_model,
            base_url=embedding_base_url,
            api_key=os.environ.get(args.embedding_api_key_env, ''),
            batch_size=args.embedding_batch_size,
        )
    rag_system = CitationRAGSystem(
        embedding_model_path=embedding_model,
        device=device,
        search_method=search_method,
        embedding_provider=embedding_provider,
        qdrant_url=args.qdrant_url or getattr(cfg, 'QDRANT_URL', None),
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
    online_manager = None
    if args.enable_per_subquery_graph:
        if search_method == 'hybrid':
            raise ValueError('Online graph rerank supports matched bm25 or vector modes, not hybrid')
        artifacts_dir = os.path.join(current_output_dir, 'online_artifacts')
        artifact_writer = ArtifactWriter(artifacts_dir, args.save_level)
        paper_db_index = load_paper_db(paper_db)
        s2_client = S2GraphClient(
            args.graph_cache_dir,
            rate_limit_rps=args.graph_rate_limit_rps,
            offline=args.graph_offline_cache_only,
        )
        scoring_backend = 'embedding' if search_method == 'vector' else 'bm25'
        processor = PerSubqueryProcessor(
            paper_db_index,
            s2_client,
            scoring_backend=scoring_backend,
            embedding_provider=embedding_provider,
            expansion_method=args.graph_method,
            expansion_limit=args.graph_expansion_limit,
        )
        online_manager = OnlinePerSubqueryManager(processor, artifact_writer, os.path.basename(current_output_dir))
        artifact_writer.write_json('run_manifest.json', {
            'upstream_repository': 'https://github.com/shenhao-stu/ScholarGym.git',
            'baseline_commit_mirror': 'https://github.com/qyx687/ScholarGym.git@baseline-repro',
            'upstream_commit': 'f426fd15e3ff28ee11ddeafc253dffd73ef88500',
            'artifact_schema_version': '1.0',
            'rank_metric_schema_version': RANK_METRIC_SCHEMA_VERSION,
            'package_source_sha256': package_source_sha256([
                'api.py', 'deeprag.py', 'eval.py', 'graph_methods.py', 'metrics.py',
                'online_per_subquery.py', 'rag.py', 'utils.py',
                os.path.join('agent', 'selector.py'),
                os.path.join('mcp', 'retrieval_mcp.py'),
            ]),
            'save_level': args.save_level,
            'config_path': args.config,
            'paper_db_path': paper_db,
            'benchmark_jsonl_path': benchmark_jsonl,
            'bm25_path': bm25_path if search_method == 'bm25' else None,
            'llm_model': llm_model,
            'prompt_type': prompt_type,
            'max_iterations': max_iterations,
            'browser_mode': browser_mode,
            'method': 'online_per_subquery',
            'search_method': search_method,
            'scoring_backend': scoring_backend,
            'embedding_backend': args.embedding_backend if embedding_provider else None,
            'embedding_model': args.embedding_service_model if embedding_provider else None,
            'embedding_base_url': embedding_provider.base_url if embedding_provider else None,
            'qdrant_url': args.qdrant_url or getattr(cfg, 'QDRANT_URL', None),
            'qdrant_collection': args.qdrant_collection,
            'graph_method': args.graph_method,
            'graph_cache_dir': args.graph_cache_dir,
            'graph_expansion_limit': args.graph_expansion_limit,
            'graph_rate_limit_rps': args.graph_rate_limit_rps,
            'graph_offline_cache_only': args.graph_offline_cache_only,
            'date_policy': 'seeds_trust_retriever_expanded_require_db_date_lte_cutoff',
            'results_per_query': results_per_query,
            'run_label': args.run_label,
            'limit': args.limit,
            'feature_weights': processor.weights,
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
        online_per_subquery_manager=online_manager,
    )
    
    # Load and process benchmark data
    benchmark_data = evaluator.load_benchmark_data(benchmark_jsonl)
    if args.limit is not None:
        benchmark_data = benchmark_data[:max(0, args.limit)]
    
    detailed_results_file = os.path.join(current_output_dir, 'detailed_results.jsonl')
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
        'PACKAGE_METHOD': 'online_per_subquery_graph_rerank' if args.enable_per_subquery_graph else 'baseline',
        'RANK_METRIC_SCHEMA_VERSION': RANK_METRIC_SCHEMA_VERSION,
        'SAVE_LEVEL': args.save_level,
        'ENABLE_PER_SUBQUERY_GRAPH': args.enable_per_subquery_graph,
        'GRAPH_METHOD': args.graph_method,
        'GRAPH_EXPANSION_LIMIT': args.graph_expansion_limit,
        'GRAPH_RATE_LIMIT_RPS': args.graph_rate_limit_rps,
        'RERANK_FEATURE_WEIGHTS': processor.weights if args.enable_per_subquery_graph else None,
        'EMBEDDING_BACKEND': args.embedding_backend if embedding_provider else None,
        'EMBEDDING_MODEL': args.embedding_service_model if embedding_provider else None,
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
