## Milestone 1，先把主链路跑通

- 目标：形成最小可运行闭环

- 要做:
    - 统一 KG schema
    - 修复 main_collect.py
    - 修复 main_experiment.py
    - 跑通：
        - Yahoo Finance OHLCV, Tavily news, KG 写入, KG 查询, decision, backtest

- 交付物
    - data/raw/... 原始数据
    - Neo4j 中可查询的节点和关系
    - trades.parquet
    - 一次完整实验日志

- 完成标准
    - 3 个 ticker 可完整跑完
    - 不靠手工 patch
    - 输出决策和收益结果

───

## Milestone 2，补齐 dissertation 最小研究系统

- 目标：让系统和 proposal 对齐，不只是 demo

- 要做: 
    - 补 FundamentalAgent
    - 补 RiskAgent
    - 强化 PortfolioManager
    - 把 signal 聚合逻辑从简单投票升级为：
        - bullish/bearish weighted aggregation
        - conflict handling
        - abstain / hold 条件更清晰

- 交付物:
  - 4 类 agent 输出
  - 每个 decision 都有 evidence refs
  - 可解释的 decision trace

- 完成标准: 
  - 每次决策都能追溯到 KG 中的 signals/evidence
  - technical/news/fundamental/risk 都参与决策

───

## Milestone 3，做真 dynamic KG

- 目标：体现 proposal 的创新点

- 要做:
  - 给 signal/evidence 加时间有效性
  - 实现新旧信息并存
  - query 按 as_of_date 过滤
• 显式记录冲突信号
• 区分 static KG 和 dynamic KG 路径

交付物

• dynamic update 逻辑
• static vs dynamic 两套实验入口
• 冲突案例样本

完成标准

• 能证明同一 ticker 在不同日期看到的 graph state 不同
• 能比较 static KG 和 dynamic KG

───

Milestone 4，补 baseline 和 evaluation

目标：形成论文核心实验

要做

• 做 3 个 baseline
  • no KG no evidence
  • evidence no KG
  • static KG
• 跑 proposed system
• 输出评价指标
  • directional accuracy
  • mean return
  • win rate
  • n trades
  • update/query latency
• 补 grounding quality 的简化评估

交付物

• 对比结果表
• case study
• ablation 草版

完成标准

• 能回答 “为什么 KG 更好”
• 能写进 results / evaluation 章节

───

Milestone 5，论文材料化

目标：从工程项目变成 dissertation

要做

• 导出图示
• 总结方法图
• 写 case study
• 写 limitations
• 写 future work

交付物

• methodology 图
• experiment tables
• example KG snapshots
• draft-ready notes

完成标准

• 论文正文可直接引用项目产出
