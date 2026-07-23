# Online 历史结果公式来源

比较或合并 Online Per-Subquery 结果前，必须先检查其 manifest 中记录的公式。
生成后的 JSONL 保持原样；run manifest 和 evaluation summary 是判断公式的依据。

| 结果目录 | 公式 | 状态 |
|---|---|---|
| `eval_results_persubquery_online_dense_litsearch_seed20260709_50_full` | `0.40 * q_norm + 0.60 * subq_norm`（`q040_sq060_closed_pool_minmax_v1`） | 已完成的历史运行；旧公式且使用修正前的 plain title+abstract 本地 rerank 输入 |
| `eval_results_persubquery_online_dense_pasa_realscholar_full` | `0.30 * q_norm + 0.40 * subq_norm + 0.15 * intent_score + 0.15 * path_count_norm` | 四项权重正确，但使用修正前的 plain title+abstract 本地 rerank 输入；旧 manifest 没有 formula ID |
| `eval_results_persubquery_online_dense_litsearch_seed20260709_50_fourfactor_full` | `0.30 * q_norm + 0.40 * subq_norm + 0.15 * intent_score + 0.15 * path_count_norm`（`q030_sq040_intent015_path015_closed_pool_minmax_v1`） | 四项权重正确，但仍是序列化修正前的历史运行；若要求严格匹配 baseline，必须新目录重跑 |

当前运行时代码只保留一个固定公式：

```text
q030_sq040_intent015_path015_closed_pool_minmax_v1
= 0.30 * query_score_normalized
+ 0.40 * subquery_score_normalized
+ 0.15 * intent_score
+ 0.15 * path_count_normalized
```

每个新实验必须使用新的输出目录和 run label。禁止把历史 litsearch50
`q040_sq060` checkpoint 续跑到当前四项式，也禁止把 embedding 序列化修改前的
dense checkpoint 与新运行混合。

当前严格 baseline 序列化策略为：

```text
scholargym_baseline_title_newline_space_abstract_v1
= "title: <title>\n abstract: <abstract>"
```

## Dynamic rerank v1（2026-07-19 起）

当前代码默认请求 `dynamic_rerank_v1`，上面的四因子公式保留为
`--no-dynamic_rerank` 静态对照以及逐 query 安全 fallback。动态模式不存在一组
全局固定数值权重；每个原始 query 的有效公式必须以该 run 的
`online_artifacts/query_rerank_policies.jsonl` 为准。manifest 中应有：

```text
dynamic_rerank_requested = true
rerank_formula_id = dynamic_rerank_v1
rerank_catalog_version = v1
rerank_prompt_version = v3
```

正式静态/动态比较必须分别使用新 output/run label，并检查比较脚本输出中的
`comparability.comparable_config=true`。动态 run 中若某个 query 的
`used_fallback=true`，该 query 实际使用旧四因子公式，不能计作动态 policy 成功。

候选论文类型证据固定为 S2 原生 positive-only `publicationTypes`。manifest 中
应记录 `paper_type_backend=s2` 与 `paper_type_namespace=s2_native`；Qwen 只生成
query policy，不再判断候选类型。类型规则只有 `prefer`、`avoid`、`exclude`；
正向强约束使用 `prefer`，`exclude` 直接由原生标签集合命中触发，不经过可配置
阈值。缺失 S2 标签表示 unknown，不构成负证据。
