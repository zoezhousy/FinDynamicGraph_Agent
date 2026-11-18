"""Explicit research CLI: real model runs, authorized review, and synthetic checks."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import json
import os
import pandas as pd


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    public_news = modes.add_parser("hsi-public-news", help="Collect a small official issuer seed corpus")
    public_news.add_argument("--output-dir", type=Path, required=True)
    relation_trial = modes.add_parser("hsi-relation-trial", help="Bounded chronological business-relation trial")
    for field in ("evidence", "market", "output-dir"):
        relation_trial.add_argument("--" + field, type=Path, required=True)
    batch = modes.add_parser("hsi-batch", help="Run full forecasts with bounded stock-level parallelism")
    for field in ("universe", "market", "evidence"):
        batch.add_argument("--" + field, type=Path, required=True)
    batch.add_argument("--output-dir", type=Path, default=Path("data/experiments/hsi"))
    batch.add_argument("--run-id", required=True)
    batch.add_argument("--workers", type=int, default=4)
    batch.add_argument("--resume", action="store_true")
    batch_replay = modes.add_parser("hsi-batch-replay")
    batch_replay.add_argument("run_dir", type=Path)
    prepare = modes.add_parser("hsi-prepare", help="Prepare official pre-test universe and daily prices")
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--news-files", type=Path, nargs="*", default=[])
    news = modes.add_parser("hsi-news", help="Collect authorized Tushare historical news")
    news.add_argument("--output", type=Path, required=True)
    news.add_argument("--start", default="2023-01-01")
    news.add_argument("--end", default="2025-12-31")
    news.add_argument("--sources", nargs="+", default=["sina"])
    hsi = modes.add_parser("hsi", help="Generate daily next-session HSI forecasts")
    for field in ("universe", "market", "evidence"):
        hsi.add_argument("--" + field, type=Path, required=True)
    hsi.add_argument("--output-dir", type=Path, default=Path("data/experiments/hsi"))
    hsi.add_argument("--run-id")
    hsi.add_argument("--resume", action="store_true")
    hsi.add_argument("--validate-only", action="store_true")
    hsi.add_argument("--tickers", nargs="+")
    hsi.add_argument("--dates", nargs="+")
    hsi.add_argument("--variants", nargs="+", choices=["full", "article_only", "no_sentiment", "single_agent"])
    hsi.add_argument("--retrieval", choices=["direct", "rules", "trace"], default="trace")
    hsi.add_argument("--response-token-budget", type=int, default=8192)
    hsi_replay = modes.add_parser("hsi-replay")
    hsi_replay.add_argument("run_dir", type=Path)
    freeze = modes.add_parser("freeze-predictions", help="Export original four-role forecasts from a research run")
    freeze.add_argument("run_dir", type=Path)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--system", default="full_graph")
    prediction_review = modes.add_parser("prediction-review", help="Review frozen next-session forecasts")
    prediction_review.add_argument("--predictions", type=Path, required=True)
    prediction_review.add_argument("--market", type=Path, required=True)
    prediction_review.add_argument("--output-dir", type=Path, required=True)
    prediction_review.add_argument("--run-id")
    prediction_review.add_argument("--resume", action="store_true")
    review_replay = modes.add_parser("prediction-review-replay", help="Replay forecast review offline")
    review_replay.add_argument("run_dir", type=Path)
    review = modes.add_parser("historical-review")
    review.add_argument("run_dir", type=Path)
    review.add_argument("--output-dir", type=Path, required=True)
    risk = modes.add_parser("risk-benchmark")
    risk.add_argument("--output-dir", type=Path, required=True)
    risk.add_argument("--repetitions", type=int, default=20)
    perturbation = modes.add_parser("evidence-perturbation")
    perturbation.add_argument("run_dir", type=Path, help="An explicitly authorized research_v1 run")
    perturbation.add_argument("--cases", type=Path, required=True, help="JSON list of independently annotated cases")
    perturbation.add_argument("--output-dir", type=Path, required=True, help="New directory outside the source run")
    multihop = modes.add_parser("multihop", help="Compare multi-hop, one-hop and lexical retrieval")
    multihop.add_argument("run_dir", type=Path, help="An explicitly authorized research_v1 run")
    multihop.add_argument("--cases", type=Path, required=True, help="JSON list of ticker/date queries and optional key evidence IDs")
    multihop.add_argument("--output-dir", type=Path, required=True, help="New directory outside the source run")
    multihop.add_argument("--top-k", type=int, default=4)
    multihop.add_argument("--seed-limit", type=int, default=4)
    multihop.add_argument("--ablate-path-rules", action="store_true", help="Also evaluate each multi-hop rule removed separately")
    multihop.add_argument("--evaluate-decisions", action="store_true", help="Rerun analysts and settle simulated trades")
    thesis = modes.add_parser("thesis", help="Run the dissertation protocol on explicit local inputs")
    thesis.add_argument("--evidence", type=Path, required=True)
    thesis.add_argument("--market", type=Path, required=True)
    thesis.add_argument("--tickers", nargs="+", default=["0005.HK", "0700.HK", "1299.HK"])
    thesis.add_argument("--train", nargs=2, metavar=("START", "END"), required=True)
    thesis.add_argument("--evaluate", nargs=2, metavar=("START", "END"), required=True)
    thesis.add_argument("--output-dir", type=Path, required=True)
    thesis.add_argument("--portfolio-mode", choices=["independent", "rolling"], default="independent")
    thesis.add_argument("--decision-dates", nargs="+", help="Override the default evaluation schedule")
    thesis_replay = modes.add_parser("thesis-replay", help="Check and recompute a dissertation run offline")
    thesis_replay.add_argument("run_dir", type=Path, help="An explicitly authorized thesis run")
    run = modes.add_parser("run")
    run.add_argument("--tickers", nargs="+", default=["0005.HK", "0700.HK", "1299.HK"])
    run.add_argument("--dates", nargs="+", required=True)
    run.add_argument("--static-cutoff", default="2025-01-01")
    run.add_argument("--source-run", type=Path, help="Use only saved successful KG task inputs from an explicitly authorized run")
    run.add_argument("--data-root", type=Path, default=Path("data/raw/market_news"))
    run.add_argument("--output-dir", type=Path, default=Path("data/experiments/research"))
    run.add_argument("--run-id")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--persist-research-graph", action="store_true", help="Write isolated research labels to configured Neo4j")
    run.add_argument("--database", default=os.getenv("NEO4J_DATABASE", "neo4j"))
    run.add_argument("--evidence-limit", type=int, default=40)
    run.add_argument("--response-token-budget", type=int, default=2048)
    run.add_argument("--risk-policy", choices=["observational", "conservative_v1"], default="conservative_v1")
    run.add_argument("--key-annotations", type=Path, help="Independent ticker:date -> key evidence IDs JSON")
    return parser


def main():
    args = build_parser().parse_args()
    if args.mode == "hsi-relation-trial":
        from src.eval.public_relation_trial import run_relation_trial
        from src.llm.client import LLMConfig, OpenAICompatibleClient
        config = LLMConfig.from_env()
        client = OpenAICompatibleClient(config)
        client.MAX_RETRIES = 1
        summary = run_relation_trial(json.loads(args.evidence.read_text(encoding="utf-8-sig")),
                                    json.loads(args.market.read_text(encoding="utf-8-sig")),
                                    client, args.output_dir)
        print(json.dumps({key: value for key, value in summary.items() if key != "predictions"}))
        return
    if args.mode == "hsi-public-news":
        from src.collectors.public_disclosures import collect_public_disclosures
        print(json.dumps(collect_public_disclosures(args.output_dir), ensure_ascii=True))
        return
    if args.mode == "hsi-batch-replay":
        from src.eval.hsi_batch import replay_batch
        print(json.dumps(replay_batch(args.run_dir)))
        return
    if args.mode == "hsi-batch":
        from src.eval.hsi_batch import run_batch
        from src.eval.thesis_sources import read_json_input
        from src.llm.client import LLMConfig, OpenAICompatibleClient
        universe, _ = read_json_input(args.universe)
        market, _ = read_json_input(args.market)
        evidence, _ = read_json_input(args.evidence)
        config = LLMConfig.from_env()
        result = run_batch(universe, market, evidence, client_factory=lambda: OpenAICompatibleClient(config),
                           output_dir=args.output_dir, run_id=args.run_id, workers=args.workers, resume=args.resume)
        print(json.dumps(result))
        return
    if args.mode == "hsi-prepare":
        from src.eval.hsi_data import prepare_hsi
        result = prepare_hsi(args.output_dir, news_files=args.news_files)
        print(json.dumps({"stocks": len(result["coverage"]), "errors": result["errors"], "news_rows": result["news_rows"]}))
        return
    if args.mode == "hsi-news":
        from src.eval.hsi_data import collect_tushare_news
        print(json.dumps(collect_tushare_news(args.output, args.start, args.end, sources=args.sources)))
        return
    if args.mode == "hsi-replay":
        from src.eval.hsi_daily import replay_hsi
        print(json.dumps(replay_hsi(args.run_dir)))
        return
    if args.mode == "hsi":
        from src.eval.hsi_daily import run_hsi, validate_inputs
        from src.eval.thesis_sources import read_json_input
        universe, _ = read_json_input(args.universe)
        market, _ = read_json_input(args.market)
        evidence, _ = read_json_input(args.evidence)
        if args.validate_only:
            tickers, tape, sources = validate_inputs(universe, market, evidence)
            print(json.dumps({"tickers": tickers, "sessions": len(tape.sessions), "bars": len(tape.prices),
                              "source_rejections": sources.rejected if sources else [], "news_available": bool(evidence)}))
            return
        from src.llm.client import LLMConfig, OpenAICompatibleClient
        result = run_hsi(universe, market, evidence, client=OpenAICompatibleClient(LLMConfig.from_env()),
                         output_dir=args.output_dir, run_id=args.run_id, resume=args.resume,
                         tickers=args.tickers, dates=args.dates, variants=args.variants,
                         retrieval=args.retrieval, budget=args.response_token_budget)
        print(json.dumps(result))
        return
    if args.mode == "freeze-predictions":
        from src.eval.prediction_review import freeze_research_predictions
        print(json.dumps(freeze_research_predictions(args.run_dir, args.output, system=args.system)))
        return
    if args.mode == "prediction-review-replay":
        from src.eval.prediction_review import replay_prediction_review
        result = replay_prediction_review(args.run_dir)
        print(json.dumps({key: result[key] for key in ("n_forecasts", "n_review_success", "n_review_failed")}, ensure_ascii=False))
        return
    if args.mode == "prediction-review":
        from src.eval.prediction_review import run_prediction_review
        from src.eval.thesis_sources import read_json_input
        from src.llm.client import LLMConfig, OpenAICompatibleClient
        predictions, _ = read_json_input(args.predictions)
        market, _ = read_json_input(args.market)
        result = run_prediction_review(predictions, market, client=OpenAICompatibleClient(LLMConfig.from_env()),
                                       output_dir=args.output_dir, run_id=args.run_id, resume=args.resume)
        print(json.dumps({key: result[key] for key in ("run_id", "run_dir", "n_forecasts", "n_review_success", "n_review_failed")}, ensure_ascii=False))
        return
    if args.mode == "thesis":
        from src.eval.thesis import run_thesis_command
        run_thesis_command(args)
        return
    if args.mode == "thesis-replay":
        from src.eval.thesis import replay_thesis
        print(json.dumps(replay_thesis(args.run_dir), ensure_ascii=False))
        return
    if args.mode == "historical-review":
        from src.eval.research_reporting import historical_review
        review = historical_review(args.run_dir, args.output_dir)
        print(json.dumps({k:review[k] for k in ("n_tasks", "status_counts", "n_successful_settled", "replay_checked", "replay_errors", "case_count")}, ensure_ascii=False))
        return
    if args.mode == "risk-benchmark":
        from src.eval.risk_benchmark import run_risk_benchmark
        df = run_risk_benchmark(args.output_dir, repetitions=args.repetitions)
        print(df.groupby(["scenario", "policy", "action"]).size().to_string())
        return
    from src.llm.client import LLMConfig, OpenAICompatibleClient
    from dataclasses import replace
    if args.mode == "multihop":
        from src.eval.research_multihop import run_multihop
        analyst = None
        if args.evaluate_decisions:
            config = LLMConfig.from_env()
            analyst = OpenAICompatibleClient(replace(config, model=os.getenv("RESEARCH_ANALYST_MODEL") or config.model))
        cases = json.loads(args.cases.read_text(encoding="utf-8"))
        rows = run_multihop(args.run_dir, args.output_dir, cases, top_k=args.top_k,
                           seed_limit=args.seed_limit, analyst_client=analyst,
                           ablate_path_rules=args.ablate_path_rules)
        print(json.dumps({"cases": len(cases), "retrieval_rows": len(rows)}))
        return
    if args.mode == "evidence-perturbation":
        from src.eval.research_perturbation import run_evidence_perturbation
        config = LLMConfig.from_env()
        client = OpenAICompatibleClient(replace(config, model=os.getenv("RESEARCH_ANALYST_MODEL") or config.model))
        cases = json.loads(args.cases.read_text(encoding="utf-8"))
        rows = run_evidence_perturbation(args.run_dir, args.output_dir, cases, analyst_client=client)
        print(json.dumps({"rows": len(rows), "successful": sum(row["status"] == "success" for row in rows)}))
        return
    from src.eval.research import ResearchConfig, run_research
    cfg = ResearchConfig(output_dir=args.output_dir, run_id=args.run_id, resume=args.resume, static_cutoff=args.static_cutoff,
                         evidence_limit=args.evidence_limit, response_token_budget=args.response_token_budget, risk_policy=args.risk_policy)
    # No dotenv file reads: credentials must already be in the process environment.
    llm_config = LLMConfig.from_env()
    clients = [OpenAICompatibleClient(replace(llm_config, model=os.getenv(f"RESEARCH_{role}_MODEL") or llm_config.model))
               for role in ("ANALYST", "CONSTRUCTOR", "VERIFIER", "JUDGE")]
    annotations = json.loads(args.key_annotations.read_text(encoding="utf-8")) if args.key_annotations else None
    with ExitStack() as owned:
        def connect(cls):
            required = ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD")
            missing = [name for name in required if not os.getenv(name)]
            if missing:
                raise RuntimeError("Missing environment variables: " + ", ".join(missing))
            client = cls(os.environ["NEO4J_URI"], os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"], database=args.database)
            owned.callback(client.close)
            client._driver.verify_connectivity()
            return client
        if args.source_run:
            from src.eval.research_sources import SavedRunSource
            query = SavedRunSource(args.source_run)
            loader = query.load_ohlcv
        else:
            from src.kg.query import KGQueryClient
            query = connect(KGQueryClient)
            def loader(ticker):
                for filename in ("ohlcv_2021_now.parquet", "ohlcv_2021_2025.parquet"):
                    path = args.data_root / ticker / filename
                    if path.exists():
                        return pd.read_parquet(path)
                raise FileNotFoundError("Market input is missing for requested ticker")
        store = None
        if args.persist_research_graph:
            from src.kg.store_neo4j import Neo4jKGStore
            store = connect(Neo4jKGStore)
        df = run_research(args.tickers, args.dates, config=cfg, query_client=query, ohlcv_loader=loader,
                          analyst_client=clients[0], proposer_client=clients[1], verifier_client=clients[2], judge_client=clients[3],
                          store=store, key_annotations=annotations)
        print(json.dumps({"run_id": df.attrs["run_id"], "run_dir": df.attrs["run_dir"], "decision_rows": len(df)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
