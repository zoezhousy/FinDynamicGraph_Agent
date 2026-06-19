from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta

from dotenv import load_dotenv

from src.config import CollectionConfig
from src.kg.store_neo4j import Neo4jKGStore
from src.pipeline import AnalysisPipeline


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def cmd_analyze(args, config):
    """分析股票"""
    trade_date = datetime.fromisoformat(args.date) if args.date else datetime.now()
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    if not tickers:
        print("❌ 请提供至少一个股票代码")
        sys.exit(1)

    all_reports: list[str] = []

    with AnalysisPipeline(config) as pipeline:
        for i, ticker in enumerate(tickers):
            if i > 0:
                print("\n")

            print(f"{'='*50}")
            print(f"📈 分析 {ticker} ({i+1}/{len(tickers)})")
            print(f"{'='*50}")

            if not args.skip_collect:
                print("📥 采集数据 + 更新 KG ...")
                summary = pipeline.collect_and_update_kg(ticker)
                print(f"   OHLCV: {summary['ohlcv_rows']} 行")
                print(f"   News:  {summary['news_rows']} 条")
                print(f"   Fundamentals: {summary['fundamental_rows']} 项")
            else:
                print("⏭️  跳过数据采集，使用现有 KG 数据")

            print("🤖 运行多智能体决策 ...")
            result = pipeline.run(
                ticker,
                trade_date=trade_date,
                with_backtest=args.backtest,
                skip_collect=True,
            )

            all_reports.append(result.report_text)

            if not args.output:
                print()
                print(result.report_text)

    if args.output:
        full_report = "\n\n".join(all_reports)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(full_report)
        print(f"\n✅ 报告已保存到 {args.output}")
    else:
        print(f"\n✅ 分析完成，共 {len(tickers)} 只股票")


def cmd_cleanup(args, config):
    """清理旧数据"""
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    pwd = os.getenv("NEO4J_PASSWORD")
    if not (uri and user and pwd):
        print("❌ 需要设置 NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD")
        sys.exit(1)

    ticker = args.ticker

    # 计算 cutoff 日期
    if args.before:
        cutoff = args.before
    elif args.days:
        cutoff = (datetime.now() - timedelta(days=int(args.days))).date().isoformat()
    else:
        print("❌ 请指定 --before 或 --days")
        sys.exit(1)

    store = Neo4jKGStore(uri, user, pwd, database=config.neo4j_database)

    try:
        print(f"🧹 清理 {ticker} 在 {cutoff} 之前的数据 ...")

        if args.type in ("all", "indicators"):
            n = store.clear_old_indicators(ticker, cutoff)
            print(f"   删除旧技术指标信号: {n} 条")

        if args.type in ("all", "news"):
            n = store.clear_old_news(ticker, cutoff)
            print(f"   删除旧新闻事件: {n} 条")

        print("✅ 清理完成")
    finally:
        store.close()


def main():
    load_dotenv()
    setup_logging()

    parser = argparse.ArgumentParser(
        description="FinDynamicGraph — 基于知识图谱的多智能体股票分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # ── analyze 子命令 ──
    p_analyze = subparsers.add_parser(
        "analyze",
        help="分析股票",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m src.cli analyze 0700.HK
  python -m src.cli analyze 0700.HK --date 2025-06-01 --backtest
  python -m src.cli analyze 0005.HK,0700.HK,1299.HK
  python -m src.cli analyze 0700.HK --skip-collect -o report.txt
        """,
    )
    p_analyze.add_argument("tickers", help="股票代码，多个用逗号分隔")
    p_analyze.add_argument("--date", help="交易日期 (YYYY-MM-DD)，默认今天")
    p_analyze.add_argument("--backtest", action="store_true", help="同时跑回测")
    p_analyze.add_argument("--skip-collect", action="store_true", help="跳过数据采集")
    p_analyze.add_argument("--output", "-o", help="报告输出路径")

    # ── cleanup 子命令 ──
    p_cleanup = subparsers.add_parser(
        "cleanup",
        help="清理 KG 中的旧数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m src.cli cleanup 0700.HK --days 90              # 删除 90 天前的数据
  python -m src.cli cleanup 0700.HK --before 2025-01-01    # 删除 2025-01-01 之前的数据
  python -m src.cli cleanup 0700.HK --days 60 --type indicators  # 只删旧技术指标
  python -m src.cli cleanup 0700.HK --days 30 --type news        # 只删旧新闻
        """,
    )
    p_cleanup.add_argument("ticker", help="股票代码，如 0700.HK")
    p_cleanup.add_argument("--before", help="删除此日期之前的数据 (YYYY-MM-DD)")
    p_cleanup.add_argument("--days", help="删除 N 天前的数据")
    p_cleanup.add_argument(
        "--type",
        choices=["all", "indicators", "news"],
        default="all",
        help="清理类型: all=全部, indicators=技术指标, news=新闻 (默认 all)",
    )

    # ── 兼容：不带子命令直接当 analyze 用 ──
    parser.add_argument("tickers_compat", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("--date", help=argparse.SUPPRESS)
    parser.add_argument("--backtest", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-collect", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", "-o", help=argparse.SUPPRESS)

    args = parser.parse_args()
    config = CollectionConfig()

    # 兼容模式：不带子命令
    if args.command is None:
        if hasattr(args, "tickers_compat") and args.tickers_compat:
            args.tickers = args.tickers_compat
            cmd_analyze(args, config)
        else:
            parser.print_help()
        return

    if args.command == "analyze":
        cmd_analyze(args, config)
    elif args.command == "cleanup":
        cmd_cleanup(args, config)


if __name__ == "__main__":
    main()
