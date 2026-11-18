# Project Status and Plan

> Last reviewed: 2026-11-18 (HSI preparation and live daily direction/review pilot).
> Evidence: 313 offline tests; ten-stock price coverage audit; live two-date generation/review and offline replay.

## Current assessment

The core research system is implemented. Data collection, temporal KG construction/query, four-agent decisions, three baselines, backtesting, provenance traces, statistical tests, and a focused temporal-safety runner all exist in code.

The independent `research_v1` path implements construction/verification collaboration, claim lifecycle, semantic evaluation, controlled variants and risk/latency reporting. A live five-variant pilot ran on two Tencent decision dates. Three Hong Kong stocks were the scope of the earlier protocol. The minimum-change dissertation scope uses daily direction predictions for ten HSI constituents fixed by official weights available before 2025. Training and rule selection use 2023–2024; testing uses 2025. Independent semantic labels and live Neo4j validation remain pending.

| Area | Status | Notes |
|---|---|---|
| Collection | Implemented | OHLCV, two news sources, global news, fundamentals, parquet and optional Neo4j |
| Dynamic KG/evidence chain | Implemented | SourceDocument -> Evidence -> Claim -> Signal/Event |
| Point-in-time query | Implemented, recently strengthened | valid_from, valid_to, effective_at, publication-time filtering |
| Multi-agent decisions | Implemented | Technical, news, fundamental, risk; LangGraph default, legacy fallback |
| Four-system comparison | Implemented | no_kg_no_evidence, evidence_no_kg, static_kg, kg_dynamic |
| Backtest/evaluation | Implemented | Five-day hold, 5 bp/side; returns, grounding, Wilcoxon, McNemar |
| Temporal leakage audit | Implemented | Future/stale counts, temporal precision, snapshot fingerprints |
| Preliminary runner | Implemented; runtime result not verified here | One ticker, 4-6 event-rich dates, four systems |
| Historical full experiment | Independently reviewed: 240 tasks, 225 success, 15 degraded | 240 offline replays matched; original run is not a controlled research_v1 experiment |
| Collaboration/lifecycle | Implemented and offline tested | Independent construction/verification calls; versioned shared ledger; optional isolated Neo4j commit unverified |
| Controlled research_v1 | Live two-date pilot completed | 7/10 successful and settled; one matched date per comparison against the full graph |
| Semantic grounding | Live pilot scores and failures recorded | Three tasks still failed on resume; independent human agreement and key-evidence coverage pending |
| Risk/update latency | Synthetic policy and memory checks run | Four fixed cases; real model behavior and Neo4j/service latency pending |

## Minimum-change HSI scope and news sources (2026-11-18)

This is the current implementation priority. The expanded experiment below is a later phase.

The first reliability implementation adds `freeze-predictions`, `prediction-review`
and `prediction-review-replay` commands. They preserve original four-role direction
forecasts, independently review evidence and written-reasoning consistency, and
evaluate category-specific accuracy together with all-forecast accuracy and retention.
The reviewer receives no future prices or labels. Source and analyst quotes are
validated, failed reviews retain diagnostics, and matching-input resume preserves
earlier failed attempts. The source export includes completed analyst decisions whose
previous semantic evaluation failed and records that selection in provenance.
This is a fixed-forecast diagnostic adapted from the supervisory evaluation idea in
GroundTrader. Daily preparation and generation now use `hsi-prepare`, `hsi`,
`hsi-replay` and optional `hsi-news`. Position deferral, adaptive budgets and
historical intervention memory are later work.

The 2026-11-18 market-only pilot uses the official July 2024 pre-test Top-10 list,
XHKG sessions and Yahoo Finance unadjusted daily prices. Preparation saved 735 bars
per stock, 7,350 bars across ten stocks, covering 2023–2025 with no missing calendar
sessions. Historical news access is unavailable; the user chose market-only workflow
validation. The first daily run uses direct verified support paths. It mines rules
offline from training-only technical graphs and makes rule retrieval an explicit
alternative. Full source lifecycle continuity across daily graphs, historical news,
the complete 2025 grid and matched multi-variant inference still need validation.

On 2026-11-18, 63 focused offline tests passed for prediction review, the existing
research protocol and run records. Verification used synthetic inputs and fixture
clients; real supervision accuracy and the daily HSI experiment remain unmeasured.

- Fix ten HSI constituents using official weights published before the 2025 test period. Keep the universe fixed during evaluation. Use 2023–2024 for training and rule selection and 2025 for daily next-session direction testing.
- Keep the four technical, news, fundamental and risk analysts, existing report schema and aggregation weights. Share one frozen stock/date snapshot. Positive aggregate score predicts UP; otherwise predict DOWN. Record missing evidence and model failures separately.
- Add sentiment to the news analyst using each article and at most four related graph facts published strictly before that article. Encode positive, neutral and negative as +1, 0 and −1 and average deduplicated articles in each forecast interval. No news remains missing.
- Compare the full framework with article-only sentiment, sentiment disabled and one integrated analyst replacing the four separate calls. Match evidence, API/model and total response-token budget. Report Accuracy, Precision, Recall, F1, coverage and failures. Company profiles, additional baselines, full HSI coverage, broad component ablations and Top-10 portfolio evaluation follow validation of this direction workflow.

### Planned data-source integration

| Source | Intended use | Historical acceptance check |
| --- | --- | --- |
| yfinance | Reuse existing Yahoo Finance OHLCV collection; add `Ticker.get_news` as a supplementary news adapter | The news API returns recent items. Verify ticker association, original publication time and accessible content. Measure actual 2023–2025 coverage before including records in historical snapshots. |
| AKShare | Add an optional adapter for relevant company news through `stock_news_em`; use documented Hong Kong market endpoints for price cross-checks where available | The news endpoint documents the latest 100 items for a symbol or keyword. Test Hong Kong company-code/name matches and retain source URLs. Historical completeness requires independent verification. |
| Tushare | Add optional `news` and `major_news` adapters with bounded date-window queries to collect 2023–2025 news | News access requires separate provider permission. Check account access and per-company historical coverage first. Request article content and publication time, split saturated result windows to avoid truncation, and deduplicate overlaps. Read the token from a process environment variable. |
| HKEXnews | Retain issuer disclosures, annual/interim reports and publication/revision records as company-background evidence | Check original release time and document version; company background used for sentiment must precede the article. |

Normalize all news adapters to article/provider ID, original publisher, title, available text, source URL when supplied, target tickers, language, publication time, retrieval time, version and content hash. Distinguish full text, excerpt and title-only records in coverage and prompts. Preserve provider and original-publisher identities, verify bilingual company matches and deduplicate syndicated content across providers. Keep retrieval time separate from original publication and revision time. Records with unknown historical availability are excluded from point-in-time snapshots with a rejection reason.

Route normalized records through the existing collection service/cache and graph ingestion boundary. Keep adapters optional and credential-free until explicitly configured, close internally owned clients, and report provider failures and permission errors. Record any provider change and freeze the same accepted news corpus for every comparison. Audit news counts by stock, month and provider, plus duplicate, rejected, missing-content and missing-coverage counts. Provider access, historical availability and source coverage remain to be verified before empirical claims.

Source references: [yfinance news](https://ranaroussi.github.io/yfinance/reference/yfinance.stock.html), [AKShare news](https://akshare.akfamily.xyz/data/stock/stock_research.html), [Tushare short news](https://tushare.pro/document/2?doc_id=143), [Tushare long news](https://tushare.pro/document/2?doc_id=195).

## Expanded HSI experiment after minimum-change validation

- **Data and prediction task.** Freeze the eligible Hang Seng Index universe using records published and effective before the 2025 test window. Train and select rules in 2023–2024, then evaluate each eligible 2025 exchange session using only information available by the Hong Kong end-of-day cutoff. Label a higher next-session close `UP` and a flat or lower close `DOWN`. Require both closes in the same split. Audit existing graph sources and verify historical publication times and versions for added news and HKEX disclosures. Report missing labels separately.
- **Graph and sentiment.** Construction and verification remain separate model calls. Check exact quotations and availability before committing to the run-owned graph, then freeze one stock/date snapshot for the analysts. Compare article-only, company-profile and Top-4-fact sentiment on the same article/company pairs, using only company background published strictly before the article. Aggregate deduplicated news by stock and forecast interval. Preserve no-news and no-background states separately from neutral sentiment.
- **Multi-agent prediction.** Run four separate analyst calls over the same frozen snapshot and retrieved paths. The technical agent assesses eligible price and volume signals, the news agent evaluates events and the graph-enhanced sentiment feature, the fundamental agent assesses issuer disclosures, and the risk agent assesses uncertainty, downside and evidence conflicts. Each returns a structured stance, signed score, confidence and supporting IDs. The deterministic aggregator reuses the existing role weights of 0.35, 0.20, 0.25 and 0.20, respectively. A positive aggregate score predicts `UP`; zero or a negative score predicts `DOWN`. Keep risk flags in the trace and record unavailable predictions and failed calls explicitly. Use the same configured API and model across all LLM calls without fixing a model name in the protocol.
- **Multi-agent ablation.** Compare the four specialist calls with one integrated analyst call that returns the same four role assessments. Use the same stock/date snapshots, retrieved paths, sentiment feature, citation checks, aggregation and total response-token budget. Split the budget across the four specialist calls; assign the full budget to the integrated call. On matched successful stock/date keys, report Accuracy, Precision, Recall, F1, paired sample size and validated evidence support. Also report prompt size, actual token use where available, model-call count, latency and failures by arm. The ablation measures the combined effect of role-separated prompts and separate calls.
- **Remaining evaluation and delivery.** Compare the full model with Historical Momentum, XGBoost, article-only sentiment and T-GNN. Include the existing controlled comparisons, component ablations and paired evidence deletion. For the separate 2025 portfolio evaluation, rank the same eligible HSI candidates at the final 2024 cutoff by each method's bullish score and hold its initial equal-weight Top 10 without rebalancing. Compare with the all-eligible-stock equal-weight buy-and-hold benchmark. Record the data, rule, model and configuration identities, preserve failed attempts and verify the outputs through offline replay. The planned HSI experiment has yet to be implemented or run.

## Verified tests

Neo4j import completed and read back on 2026-11-18 under namespace
`public_relations_20261118_01`: eight ResearchClaim records (HSBC 1, Tencent 7),
four ResearchEvidence records and eight RESEARCH_LINK support edges, plus two
ResearchGraphState records. Source URLs/version metadata and exact quotations are
retained in the serialized payloads. The existing research transaction interface
was used; these are research claim/support records, not newly materialized native
Company-to-Company PARTNERED/ACQUIRED relationships. State digests, claim/evidence
payloads and support-edge counts matched the local validated input after readback.
Report: `data/experiments/hsi/neo4j_import_20261118_01/neo4j_import_report.json`.

Live shared-prefix check (2026-11-18): the same HSBC/SVB article was sent once
for extraction and once for independent verification using deepseek-flash,
thinking disabled and a 4096-token response cap. Both calls shared identical
system and source prefixes. Input cache hits were 1920/4199 (45.73%), versus
128/2913 (4.39%) in the earlier same-article pair; misses fell from 2785 to 2279.
The new verifier alone hit 1920/2200 input tokens (87.27%). Total input grew
because both schemas/task rules are shared. This is one live pair, not a repeated
controlled benchmark. The proposed relation was rejected by the endpoint-name
quote check, so this run verifies cache behavior, not successful graph admission.
Records: `data/experiments/hsi/prompt_cache_validation_20261118_163420`.

Business extraction and verification now share the `business_shared_prefix_v1`
system prompt, both fixed schemas and canonical source/context prefix. Task selection
and candidate relations are appended last; independent verifier calls and exact-quote
admission checks remain in place. The cache identity includes the version and fixed
prompt/schema digest. Offline checks cover identical prefixes and successful local
cache reuse; the initial offline revision was followed by the live check above.

2026-11-18 public-issuer trial: four real Tencent/HSBC pages were collected into
`data/experiments/hsi/public_sources_20261118_03` with raw HTML, hashes, visible
dates and next-day UTC availability for date-only sources. Exact quote-key and
endpoint-name guidance was clarified after failed real calls; HTML layout whitespace
is normalized before inference, without changing historical saved records.
`public_relations_trial_20261118_03` used deepseek-flash with thinking disabled:
eight extraction/independent-verification calls accepted one ACQUIRED and seven
PARTNERED claims; one DIVESTED candidate was rejected because its endpoint names
did not both occur in the selected quotation. Commit replay matched the saved
deltas. No statement here establishes human-validated semantic accuracy.

The business-only trial produced three path patterns from 976 stock/day training
labels. Maximum directional confidence was 0.5211 for generic claim support,
0.5409 for acquisition claims and 0.5224 for partnership claims. No rules met the
0.60 threshold, so both test stock/dates remained without eligible paths and no
prediction analyst was called. Repeated daily exposure to one continuing fact is
not an independent corporate event. Typed source-owner claim links distinguish
business categories without collapsing subsidiaries into the listed parent.
Final-trial usage was 18,923 input tokens (896 hit / 18,027 miss) and 3,578 output
tokens; its 4.73% input hit rate concerns construction/verification, not the four
analyst shared-prefix comparison. Earlier failed attempts remain separately saved.
Full ten-stock/time-period news coverage, archived source versions and qualifying
predictive rules remain pending. The expanded default offline suite passed 354 tests.

Research business-relation support now includes PARTNERED, INVESTED_IN, SELLS,
ACQUIRED and DIVESTED,
with typed Company/Product endpoints, independent extraction/verification calls,
exact-quote and publication checks, pair-specific lifecycle keys and historical
revision/expiry projection. HSI training and test snapshots can use these facts
from imported news/disclosures. The path engine carries proof for every selected
business edge, checks terminal-source association and exposes directed inverses
explicitly during traversal. Training extraction caches are run-owned and time-
bounded. Full corporate source coverage remains pending; the existing market-only
records contain no empirical corporate-relation results. The separate initial live
issuer-relation trial is documented above.
ACQUIRED and DIVESTED require an explicit completed-transaction status and
independent source verification of completion. Directions are acquirer-to-target
and disposer-to-divested-company. Partial divestment does not automatically remove
an investment claim. Planned/approved/pending/cancelled deals are excluded from
these completed-event relations. Source publication and effective times remain
separate, and legacy business payloads retain their original serialization.
After adding ACQUIRED and DIVESTED, the default offline suite passed 348 tests. Forty focused
tests covered business relations, TRACE selection, daily forecasts and review,
including quote rejection, endpoint types/direction, expiry/revisions, coexistence
of multiple products, extraction-cache reuse and proof for intermediate edges.

The HSI entry point and full batches now select TRACE-inspired rule-only contexts.
Shared-prefix hardening canonicalizes nested context keys and business entity
ordering, and places news-only sentiment after the role task. The 53 focused
offline checks for TRACE paths, business relations, daily prediction and review
passed; no live API cache-hit improvement has been measured for this change.
The mining/search policy and input projection are versioned as `trace_paths_v1`.
Only selected source quotes and path facts enter analyst and review prompts, with
shared fixed system/schema prefixes. Training uses 2–4-hop patterns, both direction
heads, support at least five and confidence at least 0.60; search uses beam eight,
depth three, up to five source paths and confidence strictly greater than 0.60.
Company-rooted traversal, bounded sampling and heuristic selection are recorded
adaptations. Empty rule results remain failures without paid analyst calls.
Current technical-only graphs encode every observation with the same
HAS_CLAIM/SUPPORTED_BY pattern. The 2023–2024 pooled training set contains 2,356 UP
and 2,524 DOWN labels (4,880 total), giving this unconditional path a maximum
confidence of 51.72%. More discriminating, point-in-time typed state relationships
or verified event relationships are needed before qualifying rules can emerge.
Any feature/threshold selection must use internal 2023–2024 validation, retaining
2025 as the independent test set. Changing the code and model requires a new run.

Full market-only prediction `hsi_full_20261118` was started on 2026-11-18 after
the user's explicit request. It covers ten fixed stocks and all 246 sessions in
2025, with 2,460 planned full-variant forecasts and 9,840 analyst calls before
retries. Ten stock-level workers use the same configured DeepSeek model, direct
support retrieval and common response budget. The batch retains failed attempts,
retries failures once per stock and replays completed shards before aggregating.
The run was stopped after repeated provider HTTP 402 (insufficient balance), following
HTTP 429 rate-limit responses. Saved task records contain 21 successes and 32 failures;
2,407 tasks remain unattempted. The 2,460-task evaluation is incomplete. Restore
provider balance, then resume with the same run ID and fewer workers (two recommended).
Final coverage and classification scores are pending. Historical news remains
unavailable. Six focused daily/batch tests passed, including concurrent
shards, incomplete stock coverage, failure retries and aggregate offline replay.

On 2026-11-18, `hsi_market_pilot_20261118` generated full four-role forecasts
for Tencent on 2025-01-02 and 2025-01-03 using the configured DeepSeek API with
thinking disabled. Both predicted DOWN and both next-session close labels were
DOWN. `hsi_market_review_20261118` completed two valid independent reviews;
both classifications were insufficient. Both runs passed offline replay. Training
supplied 488 valid next-session labels, but no technical path pattern reached
the existing mining thresholds. Retrieval was explicitly configured as direct
support. News and company background were missing. This is market-only workflow
validation on two dates and supplies no estimate of general predictive advantage.
The default offline suite passed 313 tests, and the final focused 18 tests verify
daily generation, fixed-forecast review, matched comparisons and input boundaries.

On 2026-11-17, the live `research_deepseek_e8_b8192_20261117` run used
authorized saved Tencent snapshots for 2025-03-01 and 2025-04-01, DeepSeek
V4 Pro with thinking disabled, eight evidence items per snapshot and a common
8,192-token analyst response cap. After one resume pass, successes by variant
were 1/2 full graph, 2/2 matched text, 1/2 no grounding, 1/2 single agent and
2/2 static graph. All seven successes settled. Three tasks failed strict
semantic validation on both attempts, and the four earlier failed attempts
were retained. Offline replay completed successfully. On the one date with
all five variants settled, the four dynamic variants bought and returned
−12.73% net; the static variant abstained and had zero decision-level return.
The saved source has no news on these dates. The pilot demonstrates execution
and exposes a judge reliability limitation. Broader effects remain unmeasured.

On 2026-11-12, a close-to-close reanalysis of the historical four-system run
found 15 common successful, settled stock-date cases with both closes, including
8 rises. The dynamic system predicted 5 of those rises (62.50% UP recall).
This is a post-hoc direction diagnostic on saved actions. On one synthetic
graph with three predefined key evidence items, the full retriever recovered
three, either single-rule ablation recovered two, and direct-only and lexical
retrieval each recovered one. Real multi-hop relevance remains unmeasured.

The Python 3.11 offline suite passed **287 tests** after adding multi-hop
retrieval. A synthetic frozen graph demonstrated both two-hop conflict
traversal and three-hop shared-source traversal. The saved-snapshot test
verified paired retrieval and simulated decisions with fixture analysts.
Real retrieval relevance and decision effects still require independent
annotations and an authorized real research run.

On 2026-11-09, the full Python 3.11 offline suite passed **285 tests**. The
controlled experiment now seeds static and dynamic variants from one verified
ledger, preserves eligible technical and fundamental fields in both
representations, and reports paired effect sizes. The evidence perturbation CLI
was verified with synthetic saved snapshots and a fixture analyst. Real model
perturbation results and external integrations are pending.

On 2026-10-12, **282 offline tests passed (22.29 seconds)**, including 30 research
regressions. No integration tests or new dependency installs.

On 2026-09-26, the offline core suite completed with **166 passed**. It covers temporal filtering/audits, grounding, evidence chains, decision traces, baseline time filtering, retrieved evidence, preliminary schema, experiment summaries, and significance tests.

At the original review, repository-wide pytest did not collect cleanly because the available .venv lacks scipy, LangChain packages, and tqdm; two Neo4j files connect during import; and the Alpha Vantage file assumes a credential during import. The refactor below isolates external tests and verifies the default offline suite. External-service integration remains unverified.

## Recently completed temporal-safety work

- Added effective_at to temporal graph models and persistence.
- Historical queries no longer misuse the current is_active flag.
- Agents and snapshot summaries share one canonical load_subgraph() path.
- Added future/stale record audits and snapshot fingerprints.
- Added a direct_llm/text_rag/static_kg/kg_dynamic preliminary runner.
- Preliminary runs fail on future retrieval or identical dynamic snapshots.

This strengthens the central claim: each decision must use only information available on that date, and dynamic graph states must actually change over time.

## Current acceptance work

1. Implement the planned HSI direction protocol, audit and freeze its historical inputs, and run a small matched multi-agent pilot before the 2025 evaluation. Preserve historical runs and report failures and matched sample counts.
2. Supply independent key-evidence and atomic-unit annotations, validate judge agreement, and inspect source/period/number support. The historical case arithmetic diagnostic already demonstrates why ID matching alone is insufficient.
3. Validate optional Neo4j Research transactions and real construction/verification/update/query latency against a test database.
4. Obtain supervisor confirmation of the planned HSI scope, verify disclosure archives, availability and revisions, then assess sample size and dependence by stock and date.
5. Validate clean declared dependency versions and remote CI; present historical, synthetic and new-protocol results separately.

## Historical roadmap (original priorities retained for context)

The dated implementation and acceptance records below supersede earlier pending statuses where evidence now exists.

### P0 - Original dissertation priorities

1. **Validate the focused experiment**

   Run:

   ~~~bash
   python -m src.eval.run_preliminary --ticker 0700.HK
   ~~~

   Accept only if all four systems return rows, future_count is zero, temporal_precision is 1 where data exists, dynamic hashes change, and the static hash remains fixed.

2. **Run and freeze the full experiment**

   ~~~bash
   python -m src.main_collect
   python -m src.main_experiment
   ~~~

   Current configuration is three HK tickers, monthly dates from 2025-01-01 to runtime current date, four systems, next-period execution, five trading-day hold, and 5 bp per side. Record the commit, model, environment, date range, and data snapshot before treating results as final.

3. **Produce the result package**

   - Four-system comparison and paired significance table.
   - Ablation interpretation: evidence, KG structure, and dynamic update.
   - Two or three auditable decision-trace case studies.
   - Temporal-divergence figure/table from snapshot diffs.
   - Limitations based on observed results.

4. **Add semantic grounding evaluation**

   Implement faithfulness and hallucination rate with a fixed, structured, cached judge rubric. Keep deterministic citation/coverage metrics as the reproducible primary layer and manually label a small sample to estimate agreement.

### P1 - Engineering and methodology cleanup

1. **Wire claim lifecycle into ingestion.** Conflict, supersession, and expiry helpers exist, but the main pipeline does not fetch comparable claims, persist the new relations, or retire superseded claims.
2. **Complete technical evidence chains.** Source review on 2026-10-12 confirms the main collection path already ingests MA, EMA, MACD, RSI, Bollinger Bands and ATR via TechnicalAgent. It currently writes signal entities/relations without the complete Evidence/Claim chain; the older MA20-only builder is not the main collection path.
3. **Finish performance/risk metrics.** Preliminary retrieval and total latency exist; KG update latency, full-run aggregates, and hold/abstain under weak/conflicting evidence remain.
4. **Make tests reproducible — offline portion completed 2026-10-04.** Offline and integration tests are isolated, collection networking is blocked, and both commands are documented. Real external checks and remote CI execution remain to be verified.
5. **Run the identity probe.** Code/tests exist, but a real report was not verified in this review.

### P2 - Optional

- HKEX report/PDF collection.
- Bullish/bearish researcher agents and leave-one-agent-out ablation.
- US-market transfer experiment.
- More realistic slippage, liquidity, and position sizing.

Do not let P2 delay the P0 result package.

## Recommended sequence

1. Repair dependencies, start Neo4j, rerun offline and integration tests separately.
2. Run the 4-6-date focused experiment and prove zero leakage plus changing snapshots.
3. Freeze configuration and run the full three-ticker experiment.
4. Generate tables, significance tests, ablations, and case studies.
5. Add faithfulness/hallucination and remaining latency/risk metrics.
6. Complete claim lifecycle and technical-signal cleanup; run optional extensions only if time remains.

## Key files

| Path | Responsibility |
|---|---|
| src/main_collect.py | Collection and KG ingestion |
| src/main_experiment.py | Full experiment and exports |
| src/eval/run_preliminary.py | Focused temporal comparison |
| src/eval/temporal_audit.py | Future/stale audit and fingerprints |
| src/agents/orchestrator.py | LangGraph/legacy execution |
| src/agents/roles.py | Portfolio decision logic |
| src/kg/update_pipeline.py | Evidence-chain construction |
| src/kg/query.py | Point-in-time graph retrieval |
| src/eval/metrics.py | Metrics and significance tests |
| src/eval/grounding.py | Deterministic grounding metrics |

## Definition of dissertation-ready

- Focused run proves zero future leakage and changing dynamic snapshots.
- Full four-system run completes with frozen configuration metadata.
- Result tables and paired tests come from that frozen run.
- At least two case studies include source-to-decision provenance.
- Faithfulness/hallucination are implemented or removed from the claimed scope.
- Offline and external integration test boundaries are explicit.

# 项目重构与实验可靠性改进计划

> 修订日期：2026-10-04。状态：第一、二阶段代码及离线验收完成；第三阶段尚未实施。上文历史记录和论文 P0/P1/P2 任务继续保留，真实服务和论文实验尚未验收。

## 目标与约束

提升主流程和智能体内部的可维护性，保留现有命令、导入路径、四系统名称、输出文件及决策字段。以当前工作区为基线，保留未提交的时间安全改动。

先完成兼容性重构，再通过显式配置启用影响结果的可靠性改进，最后验证权威数据源。不同协议或数据版本的结果不得混合比较。默认不读取受保护的运行数据，不执行真实采集、模型调用或数据库清理。

## 第一阶段：兼容性重构

- 抽取采集、缓存、新闻合并和图谱写入服务；拆分实验调度、回测持久化与报告导出。旧入口保留包装函数，完整实验与 preliminary 保留各自结果结构和异常策略。
- 建立合成输入的行为基线，固定来源优先级、排序、缓存回退、OHLCV 文件选择和图谱写入顺序。共享服务通过明确参数保留不同入口的差异，避免统一实现时改变输入。
- 独立报告模型、旧版角色调用和证据辅助逻辑；确定性组合决策实现移至 src/agents/pm_deterministic.py，roles.py 保留兼容导出。保留 dataclass 与 Pydantic 两种表示，不改权重、阈值、提示词、校验和证据处理。
- 统一连接配置解析，允许注入查询客户端、存储和模型客户端。内部资源由创建者关闭，注入资源由调用者关闭；初始化失败和重复关闭安全。采集保持数据库可选，分析与实验保持数据库必需，各入口数据库选择优先级不变。
- 修复重置分支不存在的配置字段引用，使用 preserve_historical_traces；默认重置仍关闭。日志配置移至运行入口。仍被测试及示例使用的遗留实现暂时保留。
- 在原交易输出之外记录成功、降级或失败，以及失败阶段和重试次数。保留现有弃权与中性回退，但技术失败单独统计，不修改旧指标公式。
- 结构调整和缺陷修复分别验证，避免行为变化被代码移动掩盖。

## 第二阶段：可复现与时间安全协议

新增 ExperimentConfig.protocol_version，默认 legacy，改进协议为 pit_v1。保留旧入口、输出和调用方式；新运行记录单独保存。

- 每次运行生成 run_id，记录股票、日期、协议、非敏感配置、模型及后端、提示词摘要、代码版本、工作区修改状态和输入快照摘要。保存实际输入与模型响应，支持不联网重放决策、回测和报告；不记录密钥、完整环境变量或受保护文件。重放不等于再次调用 LLM 得到相同结果。
- 每个股票和决策日期共享一次加载的动态快照，工具、组合决策与审计使用同一份输入；静态系统使用固定截止时间。缓存不跨运行或股票复用。
- pit_v1 工具固定本次股票和有效截止时间，拒绝其他股票、未来日期或不同截止时间的请求，并记录原因。
- pit_v1 日期参数统一表示香港时间当日结束，再转换为 UTC 比较；发布时间、有效时间和采集时间分别保留。缺失可用时间的数据不自动视为历史可用。单独测试新闻基线与图谱日期边界差异。
- pit_v1 优先读取 ohlcv_2021_now.parquet，不存在时使用旧文件名，记录路径和摘要；legacy 保留现有入口优先级。
- 按运行、系统、股票和日期逐任务持久化；显式续跑只跳过成功任务，失败或降级任务重试。配置或快照摘要不同则拒绝续跑，新记录隔离保存，不覆盖其他运行。
- raw_return 保留现有净收益含义；新记录补充毛收益、成本、实际持仓天数与完整性。pit_v1 持仓窗口不足配置天数时记为未结算，不计入已结算收益；默认仍为五个交易日。
- 记录每个股票日期应有的四系统任务及缺失原因。新协议配对比较只使用成功、已结算且双方都有结果的任务，并报告有效样本量。

不改变角色权重、交易阈值、提示词或增加智能体，不引入新的回测框架。

## 第三阶段：权威数据源接入

对目前的三只港股，**优先接入港交所披露文件，再补官方宏观数据；行情数据视学校授权和预算决定。**

| 数据类别 | 建议来源 | 对项目的价值与限制 |
|---|---|---|
| 公司公告、年报、业绩、风险事件 | **HKEXnews 披露易** | 查询结果提供发布时间和原始文件，适合构建可追溯的证据链。已确认公开查询页面，尚未确认可直接使用的稳定公开 API。[官方查询](https://www1.hkexnews.hk/search/titlesearch.xhtml) |
| 历史行情、交易与公司行动 | **HKEX Data Marketplace** | 交易所直接提供，支持下载、SFTP 等交付。具体产品须核对字段及授权；已查的月度日终摘要列出高、低、收盘等信息，不能直接假定包含回测需要的开盘价。[官方平台](https://www.hkex.com.hk/Services/Market-Data-Services/Historical-Data-Services/HKEX-Data-Marketplace?sc_lang=en) |
| 汇率、HIBOR、货币环境 | **香港金融管理局** | 提供 JSON API，可补充风险和宏观背景。但“每日观测值”不代表每日发布，必须核对发布时间。[HIBOR 接口说明](https://data.gov.hk/en-data/dataset/hk-hkma-t06-t060303hk-interbank-ir-daily/resource/f1196f1e-e354-4520-abc7-51db11c378e2) |
| 宏观历史版本 | **DATA.GOV.HK 历史归档** | 提供文件版本查询和下载接口；具体数据集的归档覆盖仍需检查。[官方规范](https://data.gov.hk/en/help/api-spec) |

港交所说明部分历史产品有学术机构优惠，适合先询问学校图书馆或研究数据服务部门；具体产品是否适用需要确认。[官方说明](https://www.hkex.com.hk/Global/Exchange/FAQ/Market-Data/Getting-Market-Data?sc_lang=en)

接入计划增加三个要求：

- **保留原始证据**：来源、文件链接、发布时间、下载时间、文件摘要；财务指标再记录报告期间、单位及原文位置。
- **区分观测时间与可用时间**：财报期末日期不能作为市场已知日期，修订数据不能直接覆盖历史版本。
- **按实验冻结数据源**：新来源形成单独的数据版本。来源失败时明确记录，避免悄悄换成另一来源影响四系统比较。

先做 **HKEXnews 公告与财报的小范围接入验证**，强化“动态证据图谱”研究。行情暂时保留现有来源作为开发输入，拿到适合的授权数据后再独立替换。

具体实施与验收：

- 首个样本限定为 0700.HK 的业绩公告和年报；先支持官方文件及披露元数据显式导入，再验证自动获取方式，不将未确认接口作为既定 API。
- 新来源采用独立适配器，映射到现有采集和证据链接口；保留原始文档，网页或 PDF 提取逻辑不进入智能体。
- 财务指标补充币种、原文页码或位置、提取方法；提取失败明确记录，不以推测值填补。重复导入不生成重复证据，修订文件保留版本。
- 正式行情替换前核对 OHLCV、交易日、停牌、复权和公司行动；保留来源差异记录，不静默拼接不同口径。
- 来源与文件可追溯、发布时间已核验、指标可返回原文、历史版本未被覆盖，才算接入验证通过。网页和文档查阅不等同于接入成功。

官方数据能提高来源可信度，但**不能自动保证历史时间安全，也不能消除模型自身的历史知识泄漏**。

## 测试与验收

- 先运行当前离线子集，记录通过项和已有失败；历史“166 passed”不代表本次结果。
- 固定报告和合成子图比较重构前后的动作、分数、置信度、冲突、引用和追踪；覆盖四种动作及阈值边界。
- 模拟采集器、模型客户端和存储，覆盖空数据、来源失败、去重、缓存回退、文件优先级及写入顺序，使用临时文件，不读取真实运行数据。
- 验证初始化失败、运行异常、重复关闭、注入资源所有权及旧导入和命令兼容。
- 测试收集不得联网；外部检查改为测试函数和 fixtures，注册 integration 标记，默认离线运行，显式启用集成测试，凭据仅来自环境变量。
- 增加最小 Python 3.11 离线 CI，检查收集和离线测试，不提供真实服务凭据。
- 验证 pit_v1 快照一致、工具越界拒绝、香港日期边界、静态快照固定、缓存隔离、离线重放、续跑去重和不完整回测窗口。
- 数据适配器使用合成响应及文档样本验证发布时间、报告期间、单位、币种、重复导入和修订版本；真实连通性及字段覆盖单独进行集成验收。
- 离线验收只证明代码行为；真实时间覆盖、动态快照变化、外部服务和论文实验结论另行验证。

## 实施顺序与文档

1. 保存行为基线，隔离外部测试，建立离线 CI。
2. 拆分智能体内部和连接管理。
3. 抽取采集服务，拆分实验运行与导出。
4. 完成第一阶段回归，记录兼容性证据。
5. 加入运行记录、重放与 pit_v1，逐项验收。
6. 验证 HKEXnews 小范围导入；核实宏观历史版本及学校行情授权，再决定正式接入范围。
7. 更新 README 的使用与协议说明、AGENTS 的长期规则，以及 PLAN 的状态、证据和剩余工作；保留原有论文优先级，不另建状态文档。

Skill 不作为实施前置条件。流程实际通过验收后，再提炼“检查工作区 → 判断时间安全影响 → 保存行为基线 → 小步修改 → 重放与回归 → 更新 PLAN”项目 skill，引用已有规范，不复制架构或进度。本次文档更新不创建 skill。

## 参考项目与边界

- [TradingAgents](https://github.com/TauricResearch/TradingAgents)：参考历史数据约束、独立回测记录和按运行标识续跑。
- [Microsoft Qlib Recorder](https://github.com/microsoft/qlib/blob/main/qlib/workflow/recorder.py)：参考运行状态、参数、指标与产物的分离记录，不照搬环境变量或工作区内容采集。
- [FinRL](https://github.com/AI4Finance-Foundation/FinRL)：参考数据层、智能体与不同运行流程的职责划分。

以上是源码或官方文档核对，未运行外部项目或验证收益；不因此引入整套外部框架。


## 本轮实施与验收记录（2026-10-04）

- **第一阶段完成**：采集/缓存/图谱批量写入、实验调度/持久化/导出和智能体内部职责已拆分；旧入口及导入保留包装/导出。固定合成决策基线覆盖 buy、sell、hold、abstain 和阈值边界，验证动作、分数、置信度、冲突、引用与追踪。连接配置、注入资源所有权和异常关闭有离线覆盖；修复重置字段引用并移动全局日志配置。未执行数据库重置。
- **第二阶段完成**：显式 `legacy`/`pit_v1` 配置、运行身份、逐任务记录、实际组合快照摘要、模型原始/解析响应、失败阶段、续跑与历史尝试保存、离线重放、共享快照、工具边界、香港日期口径、未结算窗口和配对样本覆盖已实现。默认仍为 legacy；新协议输出按 run_id 隔离。
- **focused 验证**：使用临时合成行情和查询客户端，确认原列结构兼容，静态快照固定、动态快照变化、future_count 为零；不代表真实历史数据或论文样本已经通过。
- **离线验收**：Python 3.11.5；完整默认测试 **230 passed**（12.03 秒）；默认收集 **230 tests collected**，`pip check` 无依赖冲突，`git diff --check` 通过。初始安全子集为 145 passed；历史 166 passed 保留为历史记录。外部测试仅执行收集检查，未连接真实服务。新增 Python 3.11 离线 CI，未在远程运行。
- **环境说明**：本地补齐 scipy 1.17.1、langchain-core 1.6.6 后完成验证；现有 requirements 中对应版本为 scipy 1.16.0、langchain-core 1.5.0。本轮未改这些声明版本，不能据此宣称锁定依赖环境已验收。完整 LangChain/provider 路径采用模拟工厂测试；真实端点未调用。
- **数据边界**：未读取 .env、真实 data 运行文件或实验 CSV；未真实采集、调用模型或清理数据库，保留已有未提交修改。所有新增实验测试使用临时合成输入。
- **第三阶段待做**：HKEXnews、HKMA、DATA.GOV.HK 和授权行情接入本轮没有实施；既有规划继续保留。未创建项目 skill。

仍需独立验证：真实 Neo4j/provider 集成、锁定依赖的干净环境和远程 CI、来源历史发布时间/修订覆盖、真实 focused 与全量实验以及论文 P0 结果包。`pit_v1` 对无时区发布时间按香港时间处理、日行情使用 16:10 可用假设；这些开发口径需与正式数据源核验，不能自动证明历史时间安全或消除模型知识泄漏。


## 全量实验准备记录（2026-10-04）

用户已选择 pit_v1。三只股票的现有行情均覆盖 2021-01-04 至 2026-08-19，每只 1382 行；为避免无行情月份，拟使用 2025-01 至 2026-08 的月度日期，共 3 × 20 × 4 = 240 个任务，静态截止时间为 2025-01-01。尚未启动真实实验，也没有收益结果。

已补齐 requirements 声明的 langchain 1.3.14、langchain-openai 1.3.5、langgraph 1.2.9 及必要依赖；实际安装版本仍需在启动时冻结。导入检查、pip check 通过；更新环境后的离线回归为 230 passed（13.96 秒）。

当前进程缺少 Neo4j/LLM 环境变量。加载 .env 的前置检查被自动审批拒绝，理由是项目规则禁止读取该文件。该检查未执行；未通过其他方式绕过。等待用户明确允许应用加载 .env 仅用于服务认证，或由用户提供已配置环境变量的运行环境。真实 Neo4j 图谱覆盖、服务连通性、模型工具调用支持和正式输出仍需启动前确认。


## 全量实验启动（2026-10-04）

已按用户继续实验授权加载 .env 进行认证，未显示或记录凭据。模型 JSON 与真实工具调用预检通过；Neo4j 连接恢复。原查询在 2026-08-01 触发 Neo.TransientError.General.MemoryPoolOutOfMemoryError（事务内存上限约 716.8 MiB）。查询已改为分别聚合技术、财务、风险、新闻分支后合并，保留现有时间过滤与输出字段；230 项离线测试通过（22.34 秒），三只股票的起始/末期真实查询均通过，末期耗时约 0.84–1.14 秒。

运行 full_pit_20261004_01 已启动：pit_v1、三只股票、2025-01 至 2026-08 的 20 个逐月日期、静态截止 2025-01-01、240 个任务。结果目录为 data/experiments/runs/full_pit_20261004_01。运行状态：进行中，尚无最终比较结论。

覆盖限制：起始静态快照有技术信号，但财务、风险、新闻、证据与声明为空；2026-08-01 动态快照包含财务、风险、证据和声明，新闻仍为空。必须在最终分析披露覆盖差异，不能把不存在的新闻证据当作真实历史输入。Docker 曾为排查启动，原数据库最终由现有 Neo4j 实例提供；未删除锁文件、强制停止进程或接受新的许可证条款。


用户选择后台运行并保存最终汇总。实验进程与一次性完成检查均已启动；experiment_review.json 定期保存进度，完成后检查重放与快照变化并将验收结果追加到本 PLAN。当前仍为运行中，不把启动当作全量完成。


## 全量实验暂停（2026-10-04）

用户明确要求暂停。已停止运行 full_pit_20261004_01 的实验进程及后台检查，保留已落盘任务与配置；暂停时已记录 26/240 项，状态计数 {"success": 25, "degraded": 1}。没有删除结果或停止数据库。未落盘的在途任务需在恢复时重新执行；恢复前应核验配置、代码及已观测输入摘要。


## 全量实验恢复（2026-10-06）

用户要求继续。已核验源代码摘要、Git 状态、模型配置与原 manifest 一致，Neo4j 可连接；原运行 full_pit_20261004_01 以 resume=True 在后台恢复，保持 2025-01 至 2026-08 的原 240 项范围，不扩展到当前日期。恢复前保存 26 项（25 success、1 degraded）；成功任务按原输入摘要校验后跳过，降级与未完成任务重试。后台实验及完成检查已启动；进程标识保存在运行目录 background_processes.json，状态见 worker_state.json 与 experiment_review.json。尚未宣称全量完成。


## 全量实验后台验收结果（2026-10-06）

运行 full_pit_20261004_01：completed；任务 240/240；状态计数 {"success": 225, "degraded": 15}。

成功且已结算样本 225，已审计任务未来记录总数 0；离线重放通过 240 项，重放异常 0 项。实际样本覆盖、技术失败和快照变化见运行目录 experiment_review.json；收益及配对检验见 results_summary_latest.csv 和 significance_tests.csv（有有效比较时生成）。

该记录确认运行和技术检查状态，不自动证明模型知识无泄漏、来源历史可用性或论文结论。


## 评估口径第一批改进（2026-10-12）

- 新增显式 `evaluation_version=legacy|v2`，默认保留历史行为；v2 必须配合 `pit_v1`，通过完整实验 CLI 的 `--evaluation-version v2` 启用。运行身份、任务输入、报告和离线重放贯通版本，拒绝跨版本续跑或混合报告。旧记录缺少该字段时按 legacy 重放；未读取、迁移或覆盖已有运行。
- v2 引用精度区分缺失检索元信息与显式空检索集，并分别验证 evidence/claim ID。无引用或缺少验证依据报告 NaN，空检索集中的虚构引用精度为零；legacy 兼容路径保留。
- v2 新鲜度统一为 30 天开发窗口，覆盖有效期，区分 fresh/stale/unknown；未知引用不再计为 fresh。每个 KG 决策及 trace 保留未知数量，报告中单列汇总。类型专属窗口尚未引入，时间安全输入过滤未放宽。
- v2 方向准确率及 McNemar 使用毛收益，净收益胜率保留；各检验记录实际配对数、效应差值及 McNemar 不一致数，拒绝重复键，补充 Holm 校正字段。置信区间和时序/股票相关性处理仍待做。
- 本次默认离线回归 **252 passed（21.60 秒）**，包括 22 项新增的引用空集、类别匹配、新鲜度边界、未知时间、成本与方向区分、检验样本/校正、版本混用写出前拒绝、旧无版本记录重放和合成四系统运行/报告/重放测试。未运行真实采集、Neo4j/provider 集成或读取实验结果。本轮开始前 230 项离线测试通过，pip check 无依赖冲突。
- 核实当前安装 scipy 1.17.1、langchain-core 1.6.6 仍与 requirements 声明不同；本轮未变更依赖，干净锁定环境与远程 CI 仍需验证。
- 已修正“主采集只入图 MA20”的过时说明。现有 240 项全量实验状态仅引用本 PLAN 的历史验收记录，本轮未独立核查运行产物或收益。

后续优先项仍包括文本基线可验证引用与受控公平比较、技术证据链、声明生命周期的指标/期间级匹配及主流程接入、带时区的基本面采集和持久来源元信息、统计置信区间/相关性处理、干净依赖环境，以及语义 grounding。每项改变实验口径时继续显式版本化并使用独立 run_id；不以本轮测试通过推断论文结论成立。

本轮涉及文件的定向 git diff --check 通过。仓库整体 diff --check 仍发现 src/kg/query.py 第 68、113、218 行已有尾随空白；该文件本轮未修改，保留用户原有变更。


## 核心研究补齐与范围确认（2026-10-12）

用户已确认按**三只港股、日行情、四个分析 Agent 与确定性 PM**完成核心研究。
这是对 proposal 约 80 股、分钟行情及更多辩论角色的实现范围缩减，优先保证图协作、
可比实验和审计可行性；减少样本、市场频率及角色架构的外部有效性。
导师确认仍待完成，不能写成导师已经批准。官方披露归档、独立发布时间核验和
人工语义标注要求没有因为缩减范围而自动取消。

本轮已实施（独立 `research_v1`，不改变原四系统默认行为）：

- **图协作与生命周期**：构图角色提出原子声明，独立校验角色核对同一冻结来源，
  确定性协调器校验引用、原文片段、股票及时间后提交；按主体/谓词/指标/期间匹配，
  保留 SUPPORTED_BY、CONFLICTS_WITH、SUPERSEDES、到期和事件历史。修订不覆盖旧声明；
  历史视图隐藏后来产生的修订到期时间与失效标记。共享账本默认按运行保存，
  可选 Neo4j Research 标签原子写入、唯一约束及状态摘要并发检查已接入。
- **语义 grounding**：逐事实/推断的 supported/contradicted/not_found 判断、精确引文校验、
  faithfulness、事实幻觉率、语义引用精度、独立关键证据覆盖、带版本与响应完整性校验的
  Judge 缓存及离线评分重放。独立关键证据未标注时留空；没有伪造人工一致性结果。
- **控制变量**：full_graph 对照 text_same_evidence、no_grounding、single_agent、static_graph；
  前三项共享证据摘要、原 PM 权重/阈值及总响应 token 上限，单 Agent 也输出同样四个角色评估。
  报告调用次数、输入字符数、每组成功/失败/结算数及真实配对数。图/文本检验图表示，
  no_grounding 检验引用要求与验证的组合机制；输入 token 未完全匹配，静态组改变新鲜度。
- **延迟与风险**：记录构图、校验、提交、查询、决策的耗时；共享保守门控为无新鲜证据 abstain、
  同一事实冲突 hold，可选 observational 关闭门控观察模型。保留模型/构图失败诊断，
  续跑校验输入与配置并保存旧失败尝试。新命令、报告与重放已串通。
- **验证**：本轮新增研究回归 30 项，覆盖历史修订信息泄漏、期间/指标匹配、未知/未来/过期证据、
  存储失败不推进账本、事务及唯一约束、真假语义支持、缓存完整性、未标注覆盖、五组公平性、
  静态冻结、失败诊断/重试保存、输入漂移拒绝、离线重放和原件不改写。
  完整默认离线测试 **282 passed（22.29 秒）**。未运行 integration 测试或安装依赖。

实际结果证据分开记录：

1. **真实历史运行重新核验**：用户明确授权指定任务 JSON、manifest 和 review。
   full_pit_20261004_01 的 240 项任务：225 success、15 degraded；成功且已结算 225；
   240 项离线重放异常 0。案例包包含 1299.HK 2026-06-01 的分歧与 hold、
   0700.HK 2025-03-01 的相对收益较好案例、0700.HK 2025-04-01 的相对收益较差案例。
   原任务字节摘要前后核对一致。历史原件未迁移或覆盖。
   独立结果目录：`data/experiments/research_reviews/full_pit_20261004_01/`；
   包含 legacy 指标表、明确标为“仅指标重分析”的 v2 配对检验、三组有选择依据和同日对照的案例。
   KG 动态样本 51、静态 56、文本 58、无证据 60；实际交易数分别 44、33、18、2。
   所有有限 p 值经 Holm 校正后为 1，无显著优势证据；无证据组只有两笔交易，不能据均值排名。
   KG 决策快照缺少新闻，静态没有声明；引用存在精度不等于语义支持，旧结果不是受控消融。
2. **合成风险/延迟检查**：四种固定输入场景各 20 次计时重复，两个策略，共 160 行。
   保守策略在空/陈旧证据时 abstain、同事实冲突时 hold、正常新鲜证据时 buy；观察策略固定为 buy。
   内存提交加冻结中位约 0.432 毫秒，P95 约 0.925 毫秒，仅为本机合成内存操作。
   目录：`data/experiments/research_checks/research_v1/`。这不是 160 个独立市场样本，
   也不是模型理解风险、Neo4j 服务延迟或新协议投资收益的证明。
3. **新协议真实实验尚未运行**：当前进程没有 LLM/Neo4j 所需环境变量。
   本轮未读取 .env，未调用真实 Judge/构图模型或写入真实 Neo4j Research 节点。
   因而真实语义分数、独立人工一致性、真实受控消融效果和外部图更新延迟仍没有结果。
   代码可通过环境变量配置后的研究 CLI 执行，原始运行不能代替这次新实验。

论文验收仍需：导师范围确认、独立关键证据/事实单元人工标注与 Judge 一致性、
新协议真实运行与按成功配对数报告、真实 Neo4j 事务/延迟验收、官方来源历史发布时间
和修订归档验证，以及干净依赖环境、远程 CI 和考虑股票/时间相关性的统计推断。
不能根据 282 项离线通过或历史引用精度为 1 宣称 proposal 已完整兑现。


本轮中文结果说明：`data/experiments/research_reviews/full_pit_20261004_01/results_for_dissertation.md`。
同目录 `arithmetic_diagnostic.json` 保存 1299.HK 基本面摘要的可复查数值不一致：
摘要将 104.43 相对约 82 的目标上升空间写为约 19%，按其自身数字实际约 27.35%。
这只是确定性算术诊断，未伪装成人工标签、外部价格核验或全样本语义分数。

## Dissertation protocol implementation (2026-11-17)

The `thesis` command connects explicitly supplied local evidence and market
files, strict cutoffs and verified graph updates, Hong Kong observation
relationships, frozen 2–4-hop rule mining and beam retrieval. Mining samples up to 32 partial paths
per depth and stock-date snapshot with a recorded seed and replaces node IDs
with shared variables in the rule atoms. The workflow also supports separate
three-token probability scoring, and five-session equal-weight long/short
portfolios. Monthly independent portfolios are the default. Daily rolling
portfolios allocate five fixed capital sleeves. The `thesis-replay` command
recomputes retrieval, scores and settlement from saved run inputs.

The default offline suite passed 295 tests, including the synthetic end-to-end
workflow and historical protocol regressions. Empirical acceptance still requires authorized
point-in-time source files and a live endpoint that returns all three class
probabilities, then a real experiment with coverage and failure counts.
