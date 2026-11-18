"""Bounded issuer-relation trial with chronological construction and audit records."""
from collections import Counter
from datetime import timedelta
from pathlib import Path

from src.eval.execution import capture_execution
from src.eval.hsi_daily import analyze
from src.eval.protocol import available_time, decision_cutoff
from src.eval.records import code_identity, digest, model_identity, write_json
from src.eval.thesis_sources import LocalEvidence
from src.kg.business_relations import BusinessUpdater
from src.kg.collaboration import GraphLedger
from src.kg.thesis_paths import freeze_graph
from src.kg.trace_paths import mine_trace_rules, explore_trace
from src.sim.thesis_portfolio import MarketTape


def run_relation_trial(evidence, market, client, output_dir):
    """Four seed documents, at most two forecast dates; never weaken rule gates."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    sources = LocalEvidence(evidence)
    if sources.rejected or not 1 <= len(evidence) <= 8:
        raise ValueError("Trial requires one to eight valid source records")
    tape = MarketTape(market)
    write_json(output / 'inputs.json', {'evidence': evidence, 'market': market})
    write_json(output / 'identity.json', {'code': code_identity(), 'model': model_identity(client),
               'evidence_digest': digest(evidence), 'market_digest': digest(market),
               'protocol': 'public_relation_trial_v1',
               'scope': 'business_relations_only_not_full_hsi_experiment'})
    ledgers = {ticker: GraphLedger(ticker, strict_cutoff=True) for ticker in sources.by_ticker}
    attempts, calls = [], []
    for index, row in enumerate(sorted(evidence, key=lambda row: (row['published_at'], row['document_id']))):
        cutoff = available_time(row['published_at']) + timedelta(seconds=1)
        original = LocalEvidence([row]).snapshot(row['ticker'], cutoff)
        with capture_execution() as capture:
            try:
                result = BusinessUpdater(client, client).update(ledgers[row['ticker']], original, cutoff,
                    cache_dir=output / 'business_cache')
                attempt = {'ticker': row['ticker'], 'source': row['source'], 'status': 'completed', 'result': result}
            except Exception as exc:
                attempt = {'ticker': row['ticker'], 'source': row['source'], 'status': 'failed',
                           'error_type': type(exc).__name__}
        calls.extend(capture.calls)
        write_json(output / 'construction' / f'{index}.json', {**attempt, 'calls': capture.calls})
        attempts.append(attempt)
        if attempt['status'] == 'failed':
            # Caller can inspect preserved diagnostics and rerun under a new identity.
            break
    def snapshot(ticker, cutoff):
        return freeze_graph(ledgers[ticker], sources.snapshot(ticker, cutoff), cutoff, limit=40)
    examples = []
    for ticker in ledgers:
        for day, following in zip(tape.sessions, tape.sessions[1:]):
            if not ('2023-01-01' <= day <= following <= '2024-12-31'):
                continue
            current = tape.prices.get((ticker, day), {}).get('close')
            future = tape.prices.get((ticker, following), {}).get('close')
            if current is not None and future is not None:
                cutoff = decision_cutoff(day)
                examples.append((snapshot(ticker, cutoff), cutoff, 'UP' if future > current else 'DOWN'))
    rules, audit = mine_trace_rules(examples)
    write_json(output / 'rules.json', {'rules': rules, 'audit': audit})
    predictions = []
    for ticker in ledgers:
        day = '2025-01-02'
        cutoff = decision_cutoff(day)
        graph = snapshot(ticker, cutoff)
        context, retrieval = explore_trace(graph, cutoff, rules)
        write_json(output / f'{ticker}_snapshot.json', graph)
        write_json(output / f'{ticker}_paths.json', {'context': context, 'audit': retrieval})
        row = {'ticker': ticker, 'day': day, 'retrieval': retrieval}
        if context['paths']:
            with capture_execution() as capture:
                try:
                    forecast = analyze(ticker, day, graph, context,
                        {'status': 'not_evaluated_in_relation_trial', 'score': None}, client)
                    row.update(status='success', forecast=forecast.model_dump())
                except Exception as exc:
                    row.update(status='failed', error_type=type(exc).__name__)
            calls.extend(capture.calls)
            write_json(output / f'{ticker}_prediction_calls.json', capture.calls)
        else:
            row['status'] = 'no_eligible_rule_paths'
        predictions.append(row)
    counts = Counter(claim['predicate'] for ledger in ledgers.values() for claim in ledger.state['claims'])
    usage = Counter()
    for call in calls:
        usage.update(call.get('usage', {}))
    total_input = usage['prompt_cache_hit_tokens'] + usage['prompt_cache_miss_tokens']
    summary = {'n_sources': len(evidence), 'n_construction_attempts': len(attempts),
               'n_construction_failures': sum(row['status'] == 'failed' for row in attempts),
               'accepted_relations': dict(counts), 'mining': audit,
               'n_api_calls': len(calls), 'usage': dict(usage),
               'input_cache_hit_rate': usage['prompt_cache_hit_tokens'] / total_input if total_input else None,
               'predictions': predictions}
    write_json(output / 'summary.json', summary)
    return summary
