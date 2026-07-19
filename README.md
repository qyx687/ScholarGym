# ScholarGym OnePass 后处理实验包

Query-conditioned dynamic rerank: [experiment guide](scripts/README_dynamic_rerank.md).

这是 ScholarGym baseline 的独立实验副本，固定在上游提交
f426fd15e3ff28ee11ddeafc253dffd73ef88500。

本包只运行一次 baseline 查询轨迹，然后在不改写 baseline memory 的前提下，
对每个检索事件执行三个可选的 shadow 后处理实验：

- per_subquery：当前检索页加 S2 引用/参考文献图扩展；
- deep_event：与检索事件 offset 对齐的深检索对照；
- deep_merged：按稳定 subquery 合并预算的深检索对照。

当前 full 模式统一使用四项式：

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

Stage A 使用 --postprocess_stage materialize，只物化候选池和全部公式特征，
不执行 rerank、Top-K 或 shadow Selector，适合作为后续动态权重 Stage B 的输入。

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
