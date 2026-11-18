"""Synthetic policy check with fixed analyst reports; not market/model evidence."""
from datetime import timedelta
from pathlib import Path
import time
import pandas as pd
from src.agents.models import AgentReport
from src.agents.pm_deterministic import portfolio_manager_decide
from src.eval.protocol import decision_cutoff
from src.eval.records import digest, write_json
from src.eval.research import ROLES, risk_conditions, apply_risk_policy
from src.kg.collaboration import GraphLedger, ClaimProposal, Verification


def run_risk_benchmark(output_dir, *, repetitions=20):
    if repetitions < 1:
        raise ValueError("Repetitions must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    when = decision_cutoff("2025-02-05")
    rows, updates = [], []
    for repetition in range(repetitions):
        scenarios = {"fresh_consistent": (when - timedelta(days=1), False),
                     "empty_evidence": (None, False), "stale_evidence": (when - timedelta(days=40), False),
                     "fresh_conflicting": (when - timedelta(days=1), True)}
        for name, (availability, conflicting) in scenarios.items():
            ledger = GraphLedger("0700.HK")
            corpus = [] if availability is None else [{"evidence_id": "synthetic:e1", "ticker": "0700.HK", "source_type": "synthetic",
                     "published_at": availability.isoformat(), "valid_to": None, "text": "Revenue was 10 units; a second source reported 11 units.",
                     "content_digest": digest("Revenue was 10 units; a second source reported 11 units.")}]
            proposals = []
            for value in (["10", "11"] if conflicting else ["10"] if corpus else []):
                proposals.append(ClaimProposal(ticker="0700.HK", claim_type="fundamental", subject="revenue", predicate="reported",
                    metric="revenue", period="synthetic-period", value=value, text=f"Revenue was {value} units.", polarity="neutral",
                    evidence_ids=["synthetic:e1"], quotes={"synthetic:e1": "Revenue was 10 units; a second source reported 11 units."},
                    available_at=availability))
            reviews = [Verification(proposal_index=i, supported=True, evidence_ids=p.evidence_ids, reason="Fixed synthetic verification fixture") for i,p in enumerate(proposals)]
            started = time.perf_counter()
            ledger.commit(proposals, reviews, corpus, when)
            snapshot = ledger.freeze(when)
            updates.append({"scenario": name, "repetition": repetition, "memory_commit_and_freeze_latency": time.perf_counter()-started,
                            "accepted_claims": len(proposals), "model_calls": 0})
            refs = [e["evidence_id"] for e in snapshot["evidences"]]
            reports = [AgentReport(role, "bullish", .8, .8, "Fixed synthetic directional fixture", refs, [], []) for role in ROLES]
            base = portfolio_manager_decide("0700.HK", when, reports, snapshot, evaluation_version="v2")
            conditions = risk_conditions(snapshot, when)
            for policy in ("observational", "conservative_v1"):
                decision = apply_risk_policy(base, conditions, policy)
                rows.append({"scenario": name, "repetition": repetition, "policy": policy, "action": decision["action"],
                             **conditions, "dataset_kind": "synthetic_fixed_reports"})
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "synthetic_risk_behavior.csv", index=False)
    pd.DataFrame(updates).to_csv(output_dir / "synthetic_memory_update_latency.csv", index=False)
    write_json(output_dir / "synthetic_benchmark.json", {"dataset_kind": "synthetic_fixed_reports", "n_scenarios": 4,
               "n_repetitions": repetitions, "n_rows": len(rows), "source_archives_verified": False,
               "limitations": ["Validates deterministic risk policy and memory graph behavior only.",
                               "Repeated fixed cases are timing repetitions, not independent market samples.",
                               "No model judgment, Neo4j integration latency, or investment performance is established."]})
    return df
