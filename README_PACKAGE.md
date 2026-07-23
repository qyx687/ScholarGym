# ScholarGym Online Per-Subquery 中文运行说明

## 1. 方法概览

本目录是可独立运行的 ScholarGym baseline 副本，固定上游提交：

~~~text
f426fd15e3ff28ee11ddeafc253dffd73ef88500
~~~

在线流程：

~~~text
Planner
  -> subquery 检索一个新页面
  -> S2 citation/reference 图扩展
  -> 本地 paper DB 与日期过滤
  -> 使用与 baseline 匹配的 backend 做闭池重排
  -> 截取前 K，K 等于原始检索页实际大小
  -> 调用未修改的 baseline Selector prompt
  -> 写入 SubQueryState 和 ResearchMemory
  -> 下一轮 Planner 看到更新后的状态
~~~

图结构不会写进 Selector prompt。continue subquery 只处理本轮新检索页；之前
在线选择的论文通过 ScholarGym 原始 memory 机制影响后续迭代。

## 2. 当前公式

公式 ID：

~~~text
q030_sq040_intent015_path015_closed_pool_minmax_v1
~~~

公式：

~~~text
score = 0.30 * query_score_normalized
      + 0.40 * subquery_score_normalized
      + 0.15 * intent_score
      + 0.15 * path_count_normalized
~~~

特征定义：

- query_score_normalized：论文与原始 query 的匹配分数，在当前 seed+expanded 闭池内 min-max。
- subquery_score_normalized：论文与当前 subquery 的匹配分数，在同一闭池内 min-max。
- intent_score：只给非 seed 扩展论文赋值；methodology=1.0、result=0.75、background=0.35，多条边取最大值。
- path_count_normalized：候选在当前 seed-expanded 图中的唯一邻居数，再做闭池 min-max。

所有候选行和 run_manifest.json 都记录公式 ID、完整权重、分项特征、最终分数和排名。

最终稳定排序规则：

~~~text
rerank score 降序
-> seed 优先
-> 原始检索排名升序
-> arXiv ID 升序
~~~

### 2.1 主表指标聚合口径

对包含 N 个 benchmark query 的主表，先分别计算逐 query recall 和 precision
的 macro average：

~~~text
macro R = (R_1 + ... + R_N) / N
macro P = (P_1 + ... + P_N) / N
~~~

主表展示的 F1 再由 macro R、macro P 计算调和平均：

~~~text
F1 = 2 * macro R * macro P / (macro R + macro P)
~~~

分母为 0 时 F1 记为 0。Sel F1 使用 macro Sel R/P，若展示 Ret F1，则使用
macro Ret R/P。这里明确不使用 `(F1_1 + ... + F1_N) / N`；后者是另一种
macro-F1 定义，不能与本项目主表的 F1 混用。调和平均必须使用未四舍五入的
macro R/P 计算，最后一步才格式化为百分比。

## 3. Dense 与 baseline 的严格一致性

ScholarGym baseline 的 build_vector_db.py 使用以下精确论文字符串建 Qdrant：

~~~text
title: <title>
 abstract: <abstract>
~~~

注意 title 后是换行，abstract: 前有一个空格。Online 的：

- code/build_vector_db_configurable.py
- code/graph_methods.py 闭池 CandidateIndex

现在都使用相同格式。query/subquery 直接按原始字符串编码，不添加字段前缀。

run_manifest.json 中记录的序列化策略 ID 为：

~~~text
scholargym_baseline_title_newline_space_abstract_v1
~~~

检索 collection、在线 rerank 必须使用同一个 embedding 模型。恢复 baseline
原始 collection 是最严格的复现方式；如果重新建库，必须使用本包的可配置建库
脚本和相同模型。

## 4. 安装

~~~bash
conda create -n scholargym-graph python=3.10 -y
conda activate scholargym-graph
pip install -r requirements.txt

export DASHSCOPE_API_KEY="..."
export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export S2_API_KEY="..."
~~~

运行前自行提供：

~~~text
data/scholargym_paper_db.json
data/bm25_index.pkl
Qdrant collection（dense 模式）
~~~

## 5. BM25 运行

~~~bash
python code/eval.py \
  --config configs/config_qwen30b_api.py \
  --paper_db data/scholargym_paper_db.json \
  --benchmark_jsonl data/scholargym_bench.jsonl \
  --bm25_path data/bm25_index.pkl \
  --output_dir eval_results_online \
  --run_label bm25_q030_sq040_intent015_path015_run1 \
  --workflow deep_research \
  --search_method bm25 \
  --max_iterations 5 \
  --results_per_query 10 \
  --browser_mode NONE \
  --enable_per_subquery_graph \
  --save_level full \
  --graph_method citations_references \
  --graph_expansion_limit 100 \
  --graph_cache_dir cache/s2_graph_oracle \
  --graph_rate_limit_rps 4.0
~~~

--no-enable_per_subquery_graph 可用于 baseline-only 诊断，但正式 Online 方法应启用
图扩展。

## 6. Dense Ollama + Qdrant 运行

准备模型：

~~~bash
ollama pull qwen3-embedding:0.6b
~~~

如需重新建库：

~~~bash
python code/build_vector_db_configurable.py \
  --paper_db data/scholargym_paper_db.json \
  --qdrant_url http://localhost:6333 \
  --qdrant_collection paper_qwen3_06b \
  --embedding_backend ollama \
  --embedding_model qwen3-embedding:0.6b \
  --embedding_base_url http://localhost:11434 \
  --recreate
~~~

运行：

~~~bash
python code/eval.py \
  --config configs/config_qwen30b_api.py \
  --paper_db data/scholargym_paper_db.json \
  --benchmark_jsonl data/scholargym_bench.jsonl \
  --output_dir eval_results_online_dense \
  --run_label dense_q030_sq040_intent015_path015_run1 \
  --workflow deep_research \
  --search_method vector \
  --max_iterations 5 \
  --results_per_query 10 \
  --browser_mode NONE \
  --save_level full \
  --enable_per_subquery_graph \
  --embedding_backend ollama \
  --embedding_service_model qwen3-embedding:0.6b \
  --embedding_base_url http://localhost:11434 \
  --qdrant_url http://localhost:6333 \
  --qdrant_collection paper_qwen3_06b
~~~

## 7. OpenAI-compatible embedding API

API embedding 可以使用，但建库与运行时必须使用完全相同的服务和模型。不同模型
不是同一个实验。

~~~bash
export EMBEDDING_API_KEY="$OPENROUTER_API_KEY"

python code/build_vector_db_configurable.py \
  --paper_db data/scholargym_paper_db.json \
  --qdrant_collection paper_openrouter_qwen3_4b \
  --embedding_backend api \
  --embedding_model qwen/qwen3-embedding-4b \
  --embedding_base_url https://openrouter.ai/api/v1 \
  --recreate

python code/eval.py \
  ... \
  --search_method vector \
  --embedding_backend api \
  --embedding_service_model qwen/qwen3-embedding-4b \
  --embedding_base_url https://openrouter.ai/api/v1 \
  --embedding_api_key_env EMBEDDING_API_KEY \
  --qdrant_collection paper_openrouter_qwen3_4b
~~~

## 8. 断点续跑与公式隔离

ScholarGym 根据 detailed_results.jsonl 中已存在的 query idx 跳过完成项，因此：

- 每个公式和实验必须使用新的 run_label；
- 不要复用历史公式的输出目录；
- 不要把旧 q040_sq060 checkpoint 续跑到当前四项公式；
- embedding 序列化变化后的 dense 运行也必须使用新目录。
- 即使公式 ID 相同，序列化修正前后的结果也不能当成同一次实验续跑。

输出目录名会自动包含：

~~~text
_per_subquery_online_q030_sq040_intent015_path015
~~~

历史结果对应公式见 RESULT_FORMULA_PROVENANCE.md。

## 9. 输出

full 模式的主要文件位于：

~~~text
<run>/online_artifacts/
  run_manifest.json
  planner_events.jsonl
  raw_retrieval_rows.jsonl
  paper_rows.jsonl
  expansion_edges.jsonl
  selector_decisions.jsonl
  selector_passes.jsonl
  memory_transitions.jsonl
  query_results.jsonl
~~~

paper_rows.jsonl 每行对应一个 query/subquery/paper 候选，包含：

- seed、expanded 或二者兼有的候选类型；
- source seed/subquery 和 edge type；
- seed 的 baseline 页分数与排名；
- 所有候选的闭池 query/subquery 分数与排名；
- intent、path count；
- 公式 ID、精确权重、rerank score/rank；
- Selector 输入、选择、原因；
- 是否写入 retrieved/selected memory。

planner_events.jsonl 保存每次 Planner 调用前的结构化 ResearchMemory。
selector_passes.jsonl 保存 initial 和可选 post-browsing Selector 的输入输出。
expansion_edges.jsonl 保存每条精确 S2 扩展边及其来源、intent、edge rank、
cache/API 状态和 cutoff 上下文。

产物不会写入 API key、原始 prompt、论文标题、摘要或作者。minimal 只保留
manifest、query summary 和最终候选/选择 ID；full 保存全部中间分析行。

## 10. 排名指标

detailed_results.jsonl 保留两套相互独立的排名：

~~~text
gt_rank / avg_distance
  = 图扩展前，baseline retriever 在日期与历史选择过滤后的原始排名

local_gt_rank / local_avg_distance
  = 图扩展后，当前 seed+expanded 闭池 rerank 排名
~~~

不要把旧版本中错误写入 gt_rank 的 local rank 与当前 schema 混合。

## 11. 重要参数

~~~text
--enable_per_subquery_graph / --no-enable_per_subquery_graph
--save_level minimal|full
--run_label LABEL
--limit N
--results_per_query N
--graph_method citations|references|citations_references
--graph_expansion_limit N
--graph_cache_dir PATH
--graph_rate_limit_rps FLOAT
--graph_offline_cache_only
--embedding_backend ollama|api
--embedding_service_model MODEL
--embedding_base_url URL
--embedding_api_key_env ENV_NAME
--qdrant_url URL
--qdrant_collection NAME
~~~

两个实验进程使用同一 S2 key 时，graph_rate_limit_rps 按进程独立计算，其总和
必须保持在 key 配额内。

## 12. 测试

~~~bash
python -m pytest -q
~~~

## 13. Query-conditioned dynamic rerank v1

动态模式在每个原始 query 开始时只生成一次 policy，并在该 query 的所有
subquery、retrieval page 和 Planner iteration 中复用。动态 top-k 直接成为
Selector 输入；Selector 的 selected papers、overview 与 checklist 继续写回
`SubQueryState/ResearchMemory`，因此会影响下一轮 Planner，而不是离线后处理。

主要开关：

~~~text
--dynamic_rerank / --no-dynamic_rerank
--rerank_policy_model MODEL
--rerank_policy_cache PATH
--rerank_retry_cached_fallbacks
--rerank_min_confidence 0.60
--rerank_semantic_min_mass 0.90
--rerank_negative_weight 0.15
--rerank_max_negative_mass 0.30
--paper_type_cache PATH
--paper_type_rate_limit_rps 1.0
--paper_type_offline_cache_only / --no-paper_type_offline_cache_only
~~~

`--dynamic_rerank` 默认开启。若 policy API 失败、JSON 不合法且一次修复仍失败，
或 confidence 低于阈值，该 query 会安全回退到静态四因子公式，并在
`query_rerank_policies.jsonl` 中记录原因。policy cache key 包含
`--rerank_min_confidence`，改变阈值不会复用旧阈值下的接受结果。S2
`publicationTypes` 只在 policy 包含论文类型规则时按需解析。候选类型来源固定为
S2 原生 `publicationTypes`，规则直接使用 13 个原生标签，不再提供 Qwen/S2
backend 或 canonical/native namespace 切换。Qwen 仍用于生成 query policy，
不判断候选论文类型。类型动作只允许 `prefer`、`avoid`、`exclude`：query 中的
正向“require/must/only”约束生成高强度 `prefer`，`exclude` 则直接按原生标签集合
是否命中执行，不使用可配置的 confidence/match 阈值。S2 是 positive-only 证据：
未解析或无类型时视为 unknown，不会误触发硬过滤。若未显式给出
`--paper_type_cache`，程序使用
`cache/dynamic_rerank/paper_types_s2_native.jsonl`；loader 会拒绝旧 Qwen 缓存。

PASA-RealScholar 动态运行示例：

~~~bash
source ~/.config/hybrid-paper-graph-search/qwen.env

python code/eval.py \
  --config configs/config_qwen30b_api.py \
  --paper_db ../third_party/ScholarGym/data/hf_scholargym/scholargym_paper_db.json \
  --benchmark_jsonl ../third_party/ScholarGym/data/scholargym_pasa_realscholar.jsonl \
  --output_dir eval_results_online_dynamic_pasa \
  --run_label pasa_dynamic_rerank_v1_run1 \
  --workflow deep_research \
  --search_method vector \
  --max_iterations 5 \
  --results_per_query 10 \
  --browser_mode NONE \
  --save_level full \
  --enable_per_subquery_graph \
  --dynamic_rerank \
  --rerank_policy_cache cache/dynamic_rerank/query_policies_pasa_v1.jsonl \
  --paper_type_cache cache/dynamic_rerank/paper_types_s2_online_pasa_v1.jsonl \
  --graph_method citations_references \
  --graph_expansion_limit 100 \
  --graph_cache_dir cache/s2_graph_oracle \
  --graph_rate_limit_rps 2.0 \
  --embedding_backend ollama \
  --embedding_service_model qwen3-embedding:0.6b \
  --embedding_base_url http://127.0.0.1:11434 \
  --qdrant_url http://127.0.0.1:6433 \
  --qdrant_collection paper_knowledge_base
~~~

静态对照必须使用新的 output/run label，并把上面命令中的
`--dynamic_rerank` 改为 `--no-dynamic_rerank`。其余数据、模型、retrieval、
graph、iteration 和 browser 参数应保持完全一致。

动态 full artifact 新增或扩展：

~~~text
online_artifacts/query_rerank_policies.jsonl  # 原始输出、校验后 policy、编译权重、fallback
online_artifacts/paper_rows.jsonl             # 每篇候选的分量贡献、type evidence、filter reason
online_artifacts/selector_passes.jsonl        # Selector 输入 ID 与实际 rerank score
online_artifacts/memory_transitions.jsonl     # 写入下一轮 memory 的 ID、score 与 policy ID
online_artifacts/planner_events.jsonl         # 下一轮 Planner 输入快照与动态 policy ID
~~~

完成静态和动态运行后，可按 `query_id` 对齐并比较 candidate/Selector 的 macro、
micro F1，以及 paired bootstrap 95% CI：

~~~bash
python code/compare_online_rerank.py \
  --static_run /path/to/static/run \
  --dynamic_run /path/to/dynamic/run \
  --output /path/to/online_dynamic_vs_static.json
~~~

比较脚本会检查关键 manifest 配置及 static/dynamic mode；不满足时
`comparability.comparable_config=false`，不要把这种结果作为正式结论。
