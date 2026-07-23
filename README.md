# ScholarGym OnePass 后处理实验包

Query-conditioned dynamic rerank: [experiment guide](scripts/README_dynamic_rerank.md).
The frozen S2-native-v4 results are summarized in
[docs/s2_native_v4_results.md](docs/s2_native_v4_results.md).

这是 ScholarGym baseline 的独立实验副本，固定在上游提交
f426fd15e3ff28ee11ddeafc253dffd73ef88500。

本包只运行一次 baseline 查询轨迹，然后在不改写 baseline memory 的前提下，
执行两个可选的 shadow 后处理实验：

- per_subquery：当前检索页加 S2 引用/参考文献图扩展；
- deep_merged：按稳定 subquery 合并预算的深检索对照。

`full` 模式可显式选择静态或 query-conditioned 动态 rerank：

~~~bash
--no-dynamic_rerank                 # 默认；严格静态 baseline
--dynamic_rerank --paper_type_cache <native-s2-cache.jsonl>
~~~

动态模式固定使用 S2 原生 `publicationTypes`；Qwen 只生成 query policy。

静态公式为：

~~~text
score = 0.30 * query_score_normalized
      + 0.40 * subquery_score_normalized
      + 0.15 * intent_score
      + 0.15 * path_count_normalized
~~~

公式 ID：

~~~text
q030_sq040_intent015_path015_closed_pool_minmax_v1
~~~

动态模式在每个原始 query 开始时只调用一次 policy LLM，并把同一 policy 用于
该 query 的全部 OnePass graph event。候选论文类型固定由 S2 原生
`publicationTypes` 提供；policy LLM 只负责读取 query 并生成 rerank policy。

Stage A 使用 --postprocess_stage materialize，只物化候选池和全部公式特征，
不执行 rerank、Top-K 或 shadow Selector；该阶段不能同时启用
`--dynamic_rerank`。

## 主表指标口径

主表中的 Sel R 和 Sel P 分别是逐 query recall、precision 的 macro average。
Sel F1 不使用逐 query F1 的平均值，而是由这两个 macro 指标重新计算：

~~~text
Sel F1 = 2 * macro(Sel R) * macro(Sel P)
         / (macro(Sel R) + macro(Sel P))
~~~

分母为 0 时记为 0。若主表展示 Ret F1，也对 macro Ret R、macro Ret P 使用
同一调和平均口径。计算时使用未四舍五入的 macro R/P，最后才格式化百分比。

现有 summary 字段 `avg_selection_f1` 和 `avg_candidate_f1` 保留的是
`mean(query-level F1)`，仅供兼容和逐 query 分析，不能直接填入主表。主表应从
对应的 macro recall、macro precision 字段按上式重新计算。

安装、运行、断点续跑和产物结构见 [中文详细说明](README_PACKAGE.md)。
