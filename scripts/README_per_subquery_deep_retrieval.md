# Per-subquery BM25 deep-retrieval replay

这是保留给既有 full 运行的离线 replay 脚本；同等的 deep_merged 实验臂现已接入
`code/eval.py`，新实验优先使用 `README_PACKAGE.md` 的单次运行方式。此脚本读取一次已经完成的
OnePass `full` 运行，冻结 baseline 2nd 的 Planner、subquery、continue、
checklist、offset 和 exclusion 轨迹；不重跑 Planner、baseline Selector 或
Semantic Scholar 图拓展。

脚本：`scripts/replay_per_subquery_deep_retrieval.py`

## `merged_subquery_sum_budget_text_deep_retrieval`

按稳定的 `(query_id, subquery_id)` 分组；相同文本但新 `subquery_id` 不合并。

1. 对同一 subquery 的所有 event 令 `N = ΣN_i`。跨 event 重复出现的 graph
   论文仍重复计入预算，匹配候选处理 occurrence 数；同时另存 graph union
   大小、重复数和 `union/sum`。
2. 固定该 subquery 第一次 event 之前的 baseline exclusion snapshot，从
   offset 0 一次取 `N` 篇 BM25 论文。
3. 对整个 `N` 池只做一次闭集归一化和重排。
4. 按原 event 时间顺序，使用各 event 实际的 `selector_top_k` 切连续且互斥的
   slice：`[0:k1)`、`[k1:k1+k2)`……。
5. 每个 slice 使用对应 event 的 checklist 独立调用 Selector；
   `old_overview=""`，选择结果不反馈到下一 slice 或 Planner。

该方案不继承 continue 的跳页问题，但它改变了归一化范围和候选分配，因此是
完整方法对照，不应把最终差异只归因于候选生成。

## 重排公式

所有深检索论文的 intent/path 特征仍固定为 0 并保存在记录中，但不参与得分。
固定公式 ID 为 `q030_sq040_intent015_path015_closed_pool_minmax_v1`：

```text
0.30 * query_score_normalized
+ 0.40 * subquery_score_normalized
+ 0.15 * intent_score
+ 0.15 * path_count_normalized
```

保存的 score 尺度为 `[0, 1]`。Q/SQ 都在当前 merged subquery 池内
用 BM25 重新计算并 min-max。

## 输入要求与历史重复处理

源运行必须是 `--save_level full`，至少包含：

```text
detailed_results.jsonl
onepass_artifacts/baseline/planner_events.jsonl
onepass_artifacts/baseline/paper_rows.jsonl
onepass_artifacts/per_subquery/paper_rows.jsonl
```

`detailed_results.jsonl` 每个 `idx` 的最后一条记录是 commit source of truth。
脚本先按 iteration 编号重启切分连续 Planner trajectories，再从后向前选择同时
满足以下条件的最新完整 trajectory：retrieved/selected ID 与 detailed baseline
完全一致、每个 event 的 per-subquery candidate/selected ID 完全一致，且每个
event 都有局部 graph pool。随后只保留与该 trajectory 的 checklist/subquery
元数据匹配的最后连续 event block，并按 event 内 arXiv ID 去重。query 36 的
两套五轮历史记录因此不会混合；较新的不完整重跑也不会覆盖较早的 committed
完整轨迹。

深检索本身以 canonical arXiv ID 为论文身份，在 exclusion/offset 前规范化并
去重；每个 request 的输出 diagnostics 会明确记录该策略。

## PaSa RealScholar：完整运行

在本目录执行：

```bash
SOURCE_RUN='eval_results_onepass_pasa_realscholar/qwen3-30b-a3b-instruct-2507_complex_bm25_deep_research_topk-[5, 10, 20]_maxq-10_instruct_non-structured_NONE_bm25_qwen30b_nothink_full_run1'

python scripts/replay_per_subquery_deep_retrieval.py \
  --baseline_run_dir "$SOURCE_RUN" \
  --bm25_path ../third_party/ScholarGym/data/bm25_index.pkl \
  --config configs/config_qwen30b_api.py \
  --output_dir eval_results_deep_retrieval_pasa_realscholar \
  --save_level full
```

Selector 使用 `configs/config_qwen30b_api.py` 的 Qwen 30B API 且 no-thinking。
每个 baseline retrieval event 对应 merged pool 中的一个 Selector slice；
当前完整 PaSa 源运行预计为 `894` 次 Selector 调用。

正式付费运行前，建议先做不调用 Selector 的一条 query smoke test：

```bash
python scripts/replay_per_subquery_deep_retrieval.py \
  --baseline_run_dir "$SOURCE_RUN" \
  --bm25_path ../third_party/ScholarGym/data/bm25_index.pkl \
  --output_dir eval_results_deep_retrieval_pasa_smoke_no_selector \
  --save_level full \
  --skip_selector \
  --limit 1
```

`--skip_selector` 的输出明确把 selection 指标写为 `null`，不会误记为 0。

## LitSearch

脚本不绑定 PaSa。将 `--baseline_run_dir` 换成完成的 LitSearch full run，并使用
新的 `--output_dir` 即可；BM25 索引不变。

## 续跑与输出

每个 method/query 完成后才原子写入：

```text
run_manifest.json
evaluation_summary.json
<method>/queries/000000.json
<method>/query_results.jsonl
<method>/summary.json
```

中断后原样重输命令即可。已存在且 `run_signature` 一致的 method/query 文件会
跳过；未完成的当前 query 会整体重做，不会生成边计算边追加的重复局部
artifact。若中断发生在一个 method/query 的多次
Selector 调用中间，该 method/query 尚未 commit，续跑会重新执行它此前已完成的
Selector 调用。`--force` 只用于同一签名下主动覆盖重算。

`run_signature` 绑定源 artifact 与 BM25 的 size/mtime、replay/Selector/prompt/
BM25 相关代码 SHA256、Selector config、provider route/base URL hash 和实验语义
（不保存 endpoint 明文或 API key）。同一输出目录已有 query 文件
但签名不同会直接拒绝运行，必须换新目录；两个进程也不要并发写同一个
`--output_dir`。

`minimal` 仍保存每层完整 arXiv ID 列表、每 event/group 的预算与来源 graph
pool ID、Selector slice/selected ID 和指标。`full` 另外保存所有
`(query, subquery/event, paper)` 分析行，包括：

- full-corpus BM25 raw score、date-valid global rank、exclusion 后 rank；
- 闭集 query/subquery raw score、normalized score 和 component rank；
- text-only rerank score/rank，固定为 0 的 intent/path；
- Selector slice/event、input rank、selected 和 reason；
- source graph event、每个 `N_i`、offset/exclusion/checklist、merged
  occurrence/union 诊断；
- `candidate_type/retrieval_backend/passed_date_cutoff/is_ground_truth`、rank scope、
  `rerankable`；极少数不能闭集打分的深检索论文也保留一行，重排字段为 null 并
  写明 drop reason。

不保存渲染后的完整 Selector prompt、原始 title/abstract/author 或 API key；
为了复现实验，会保存 query、subquery、Planner checklist、Selector reason 和
overview。每个 query 和总表同时报告：

1. 完整 deep pool（候选生成）；
2. rerank 后实际送入 Selector 的 slice 并集（主表 Ret 口径）；
3. Selector selected（Sel 口径）。

总表同时保存 `mean_query_*_f1` 和
`macro_*_f1_from_avg_recall_precision`；`avg_candidate_f1`/
`avg_selection_f1` 沿用当前主表口径，即 macro recall 与 macro precision 的
调和均值。预算 requested/actual、fulfillment、rerankable/unscorable 和 source
graph occurrence/union 也直接聚合，无需重新扫描全部 paper rows。

## 内存

脚本直接复用 baseline BM25 pickle 中的 metadata，不额外加载约 820 MB 的完整
paper DB。正式 pickle 本身约 2 GB，但其中包含 Python token corpus、BM25 对象、
metadata 和 mappings；在当前 12 GB WSL 上实测反序列化阶段 RSS 可接近 10 GB。
建议至少 16 GB RAM（12 GB 机器应保留足够 swap 且会明显更慢）。加载完成后，
脚本每次只保留一个 benchmark query 的 artifact 和 compact deep pools；同一
subquery 的 full-corpus BM25 score 只计算一次。
