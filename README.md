# ScholarGym 在线 Per-Subquery 图扩展重排

这是 ScholarGym baseline 的独立实验副本，固定在上游提交
f426fd15e3ff28ee11ddeafc253dffd73ef88500。

本方法在每次 subquery 返回一个新检索页后，立即进行 S2 图扩展、闭池重排和
原始 Selector 调用，并把结果写回 ScholarGym memory。因此它会改变后续 Planner
看到的状态，是端到端在线方法，不是 shadow replay。

默认启用 query-conditioned dynamic rerank。Qwen 每个原始 query 只生成一次
离散 policy，确定性编译器在全部 subquery/iteration 中复用。候选论文类型证据
默认来自 S2，也可显式切换为 query-independent 的 Qwen title+abstract 分类器：

~~~text
--paper_type_backend s2    # default
--paper_type_backend qwen
~~~

静态对照及动态失败时的回退公式为：

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

Dense 模式的论文 embedding 输入已与 baseline Qdrant 建库严格统一为：

~~~text
title: <title>
 abstract: <abstract>
~~~

## 主表指标口径

主表中的 Sel R 和 Sel P 分别是逐 query recall、precision 的 macro average。
Sel F1 不使用逐 query F1 的平均值，而是由这两个 macro 指标重新计算：

~~~text
Sel F1 = 2 * macro(Sel R) * macro(Sel P)
         / (macro(Sel R) + macro(Sel P))
~~~

分母为 0 时记为 0。若主表展示 Ret F1，也对 macro Ret R、macro Ret P 使用
同一调和平均口径。计算时使用未四舍五入的 macro R/P，最后才格式化百分比。

安装、运行、断点和产物说明见 [中文详细说明](README_PACKAGE.md)。
历史结果对应的公式见 [结果公式来源说明](RESULT_FORMULA_PROVENANCE.md)。
