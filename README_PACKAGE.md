# ScholarGym OnePass 中文运行说明

## 1. 方法概览

本目录是可独立运行的 ScholarGym baseline 副本，固定上游提交：

~~~text
f426fd15e3ff28ee11ddeafc253dffd73ef88500
~~~

每个 benchmark query 只执行一次完整 baseline 轨迹：

~~~text
baseline Planner / Retriever / Selector
  -> 可选 per-subquery 图扩展与 shadow Selector
  -> 可选稳定 subquery 合并预算的 deep 检索与 shadow Selector（deep_merged）
  -> 下一个 benchmark query
~~~

两个 shadow 分支不会写回 baseline memory，因此不会改变 baseline 的后续
Planner 轨迹。deep_merged 的候选预算来自已经生成的 graph local pool，不会再次
调用 Semantic Scholar 来决定预算。

### 1.1 主表指标聚合口径

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

OnePass 外层 summary 当前为兼容既有产物，仍保留：

~~~text
avg_selection_f1 = mean(query-level selection F1)
avg_candidate_f1 = mean(query-level candidate F1)
~~~

这两个字段不能直接填入主表。主表 Sel F1 应从 `avg_selection_recall`、
`avg_selection_precision` 重算；若展示 Ret F1，则从相应 candidate/retrieval
macro R/P 字段重算。

## 2. 两种运行阶段

### 2.1 Stage A：只物化候选池和特征

使用：

~~~bash
--postprocess_stage materialize
~~~

Stage A 仍完整运行一次 baseline，因为 retrieval event、continue offset、
冻结 exclusion、checklist 和 Selector budget 都由 baseline 轨迹决定。
两个后处理分支只生成：

~~~text
per_subquery: 图扩展/过滤 + 闭池 Q/SQ + intent/path 特征
deep_merged:  sum-budget deep pool + 检索排名 + 闭池 Q/SQ 特征
~~~

Stage A 不计算最终 rerank score，不产生 rerank Top-K，也不调用 shadow
Selector。产物中会明确记录：

~~~text
legacy_rerank_applied=false
shadow_selector_applied=false
~~~

为了后续 Stage B 能读取完整的候选行和图边，建议使用：

~~~bash
--postprocess_stage materialize
--save_level full
~~~

Stage A 采用 query 级原子提交。某个 query 的图扩展、embedding 或 deep
检索失败时，该 query 的临时行不会进入正式 JSONL，也不会被 checkpoint；
用完全相同的命令重跑即可重试。

现有 scripts/replay_graph_rerank_formulas.py 面向历史 full-run schema，
不是 Stage A 的正式动态权重 Stage B 实现。

### 2.2 Full：静态/动态 rerank 加 shadow Selector

默认 `--no-dynamic_rerank` 使用固定公式：

~~~text
q030_sq040_intent015_path015_closed_pool_minmax_v1
~~~

公式为：

~~~text
score = 0.30 * query_score_normalized
      + 0.40 * subquery_score_normalized
      + 0.15 * intent_score
      + 0.15 * path_count_normalized
~~~

graph 分支使用全部四项。deep 分支没有图边，因此 intent_score 和
path_count_normalized 明确保存为 0；其有效排序分数为 0.30Q + 0.40SQ，
但 manifest 和候选行仍使用同一个四项公式 ID 与完整权重表。

启用 `--dynamic_rerank` 后，每个原始 query 只生成一次 policy，所有 graph
event 共用该 policy。policy 可以按 query 选择/关闭 citation intent、path count、
语义相似度和 paper-type alignment，并可用负权重表达排除倾向。动态模式只改变
`per_subquery` graph arm；`deep_merged` 继续作为固定 text-only control。

候选论文类型固定读取 Semantic Scholar 原生 `publicationTypes`，不再提供 Qwen/S2
backend 或 canonical/native namespace 切换。policy LLM 仍负责阅读原始 query、
生成维度权重和类型规则；它不判断候选论文类型。策略直接使用 `Review`、
`Conference` 等 13 个 S2 原生标签，不经过功能类型映射。S2 缺失标签按 unknown
而不是 negative 处理。

## 3. 特征定义

- query_score_normalized：论文与原始 query 的本地匹配分数，在当前闭合候选池内 min-max。
- subquery_score_normalized：论文与当前 subquery 的匹配分数，在当前闭合候选池内 min-max。
- intent_score：只给非 seed 图扩展论文赋值；methodology=1.0、result=0.75、background=0.35，多条边取最大值。
- path_count_normalized：候选在当前 seed-expanded 图中的唯一邻居数，再在当前闭池 min-max。

如果一个特征在当前池内全部相同，min-max 结果统一为 0。

最终排序的稳定 tie-break 为：

~~~text
rerank score 降序
-> seed 优先
-> baseline 观测排名升序
-> arXiv ID 升序
~~~

## 4. Dense 与 baseline 的严格一致性

baseline Qdrant 建库时，论文 embedding 输入字符串严格为：

~~~text
title: <title>
 abstract: <abstract>
~~~

也就是 title 后有换行，abstract: 前保留一个空格。OnePass 的可配置建库脚本
和闭池 reranker 都使用这个精确格式。query 和 subquery 直接按原始字符串编码，
不添加字段前缀。

run manifest 中记录的序列化策略 ID 为：

~~~text
scholargym_baseline_title_newline_space_abstract_v1
~~~

默认 dense 模型固定为：

~~~text
qwen3-embedding:0.6b
~~~

检索与 rerank 必须使用同一个 embedding 模型和同一种论文序列化。严格复现时，
优先恢复原始 Qdrant collection，而不是重新建库。

## 5. 两个后处理分支

### per_subquery

针对每个新检索页，使用页内 seed 做 citation/reference 扩展。扩展论文必须：

- 能映射到本地 paper DB；
- 具有有效月份；
- 不晚于 query/subquery cutoff；
- 不在该事件冻结的已选 exclusion 中。

### deep_merged

将相同稳定 subquery_id 的 continue 事件合并，预算为 N=sum_i N_i。从 offset 0
检索并重排一次，再按时间顺序切成互不重叠的 [0:k1)、[k1:k1+k2) 等 Selector
输入。每一片使用自己的 checklist 和 iteration，前一片的选择不会写入后一片。

## 6. 安装

~~~bash
conda create -n scholargym-graph python=3.10 -y
conda activate scholargym-graph
pip install -r requirements.txt
~~~

配置 LLM 和 Semantic Scholar 密钥：

~~~bash
export DASHSCOPE_API_KEY="..."
export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export S2_API_KEY="..."
~~~

DASHSCOPE 用于 Planner/Selector；动态模式下也用于 query policy，选择 Qwen
候选类型 backend 时还用于 title+abstract 类型分类。默认 dense embedding 由
本地 Ollama 提供。

运行前需要自行提供大文件：

~~~text
data/scholargym_paper_db.json
data/bm25_index.pkl
Qdrant paper_knowledge_base collection
~~~

## 7. BM25 full 运行示例

~~~bash
python code/eval.py \
  --config configs/config_qwen30b_api.py \
  --paper_db data/scholargym_paper_db.json \
  --benchmark_jsonl data/scholargym_bench.jsonl \
  --bm25_path data/bm25_index.pkl \
  --output_dir eval_results_onepass \
  --run_label bm25_q030_sq040_intent015_path015_run1 \
  --workflow deep_research \
  --search_method bm25 \
  --max_iterations 5 \
  --results_per_query 10 \
  --browser_mode NONE \
  --save_level full \
  --postprocess_stage full \
  --no-dynamic_rerank \
  --run_per_subquery_postprocess \
  --run_deep_merged_postprocess \
  --graph_method citations_references \
  --graph_expansion_limit 100 \
  --graph_cache_dir cache/s2_graph_oracle \
  --graph_rate_limit_rps 4.0 \
  --postprocess_event_workers 4 \
  --postprocess_selector_concurrency 4 \
  --postprocess_embedding_concurrency 1
~~~

可关闭 deep_merged 对照：

~~~bash
--no-run_deep_merged_postprocess
~~~

deep_merged 依赖 graph pool 提供预算，所以启用时必须启用
--run_per_subquery_postprocess。

动态组在上面的数据/检索参数不变时替换为：

~~~bash
--dynamic_rerank \
--rerank_policy_cache cache/dynamic_rerank/onepass_query_policies_pasa_v1.jsonl \
--paper_type_cache cache/dynamic_rerank/onepass_paper_types_s2_pasa_v1.jsonl
~~~

`--paper_type_cache` 只接受 `evidence_source=semantic_scholar` 的记录；旧 Qwen 或
canonical 类型缓存会被拒绝，避免跨实验语义污染。

## 8. Dense 运行示例

启动 Ollama 与 Qdrant，并准备模型：

~~~bash
ollama pull qwen3-embedding:0.6b
~~~

如不恢复原始 collection，可按 baseline 序列化重新建库：

~~~bash
python code/build_vector_db_configurable.py \
  --paper_db data/scholargym_paper_db.json \
  --qdrant_url http://localhost:6433 \
  --qdrant_collection paper_knowledge_base \
  --embedding_base_url http://localhost:11434 \
  --batch_size 64 \
  --recreate
~~~

运行：

~~~bash
python code/eval.py \
  --config configs/config_qwen30b_api.py \
  --paper_db data/scholargym_paper_db.json \
  --benchmark_jsonl data/scholargym_bench.jsonl \
  --output_dir eval_results_onepass_dense \
  --run_label dense_q030_sq040_intent015_path015_run1 \
  --workflow deep_research \
  --search_method vector \
  --results_per_query 10 \
  --max_iterations 5 \
  --browser_mode NONE \
  --save_level full \
  --postprocess_stage full \
  --no-dynamic_rerank \
  --embedding_base_url http://localhost:11434 \
  --qdrant_url http://localhost:6433 \
  --qdrant_collection paper_knowledge_base \
  --postprocess_event_workers 4 \
  --postprocess_selector_concurrency 4 \
  --postprocess_embedding_concurrency 1
~~~

Stage A 使用同一条命令，只需把：

~~~bash
--postprocess_stage full
~~~

替换为：

~~~bash
--postprocess_stage materialize
~~~

## 9. 断点续跑

checkpoint 以 detailed_results.jsonl 中已经提交的 benchmark idx 为准。
必须使用完全相同的命令和输出目录才能正常续跑。

run_manifest.json 的签名包含：

- 代码、配置与 benchmark 内容；
- paper DB、BM25/Qdrant/Ollama 定位信息；
- baseline 参数和两个后处理开关；
- static/dynamic、policy、候选类型 backend、公式 ID 与精确权重；
- embedding、并发和缓存策略。

公式、dynamic 开关或候选类型 backend 发生变化后不能续跑旧目录。静态、动态
S2、动态 Qwen 必须使用各自的输出目录；程序也会把 rerank/backend 写入目录名。

--allow_resume_compatible_code_change 只允许经过审计且不改变实验结果的代码升级，
不能用于公式、模型、prompt、数据集或检索参数变化。

## 10. 产物

full 模式的主要文件位于：

~~~text
<run>/onepass_artifacts/
  baseline/planner_events.jsonl
  baseline/paper_rows.jsonl
  baseline/selector_decisions.jsonl
  per_subquery/paper_rows.jsonl
  per_subquery/pool_records.jsonl
  per_subquery/expansion_edges.jsonl
  per_subquery/selector_decisions.jsonl
  deep_merged/pool_records.jsonl
  deep_merged/paper_rows.jsonl
  deep_merged/comparisons.jsonl
  deep_merged/selector_decisions.jsonl
  query_rerank_policies.jsonl
  query_results.jsonl
  run_manifest.json
  resume_reconciliation.json
~~~

minimal 保留 manifest、query 指标和紧凑 pool 记录；full 额外保存逐论文行、
精确扩展边和 Selector decision。产物不写入 API key、原始 prompt、论文标题、
摘要或作者。

Stage A 的 summary 只报告候选池覆盖、预算满足率和特征可计算数量；由于没有
rerank Top-K 和 shadow selection，candidate/selection 指标会明确保存为不可用，
而不是伪装成 0。

## 11. 重要参数

~~~text
--save_level minimal|full
--postprocess_stage full|materialize
--run_label LABEL
--limit N
--results_per_query N
--run_per_subquery_postprocess / --no-run_per_subquery_postprocess
--run_deep_merged_postprocess / --no-run_deep_merged_postprocess
--dynamic_rerank / --no-dynamic_rerank
--rerank_policy_model MODEL
--rerank_policy_cache PATH
--paper_type_cache PATH
--paper_type_rate_limit_rps FLOAT
--paper_type_offline_cache_only / --no-paper_type_offline_cache_only
--graph_method citations|references|citations_references
--graph_expansion_limit N
--graph_cache_dir PATH
--graph_rate_limit_rps FLOAT
--graph_offline_cache_only
--embedding_base_url OLLAMA_URL
--qdrant_url URL
--qdrant_collection NAME
--postprocess_event_workers N
--postprocess_selector_concurrency N
--postprocess_embedding_concurrency N
--allow_resume_compatible_code_change
~~~

同一 S2 key 被两个进程同时使用时，graph_rate_limit_rps 是按进程计算的，
两个进程的速率之和必须保持在 key 配额内。

## 12. 测试

~~~bash
python -m pytest -q
~~~
