from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import json
import pandas as pd
import pytest

from src.eval.protocol import decision_cutoff
from src.eval.records import digest, write_json
from src.kg.collaboration import GraphLedger, ClaimProposal, Verification, evidence_corpus, CollaborativeUpdater
from src.eval.semantic_grounding import SemanticEvaluator, JudgeResponse, score_judgment, annotation_agreement
from src.eval.research import ResearchConfig, run_research, frozen_view, VARIANTS, risk_conditions
from src.eval.replay import replay_record
from src.eval.research_reporting import export_research
from src.eval.research_perturbation import run_evidence_perturbation, _views
from src.eval.research_multihop import run_multihop
from src.eval.execution import record_call
from src.kg.research_paths import retrieve_research_evidence

WHEN = decision_cutoff("2025-01-02")


def source(eid="e1", text="Revenue was 10 units.", when=WHEN):
    return {"evidence_id": eid, "ticker": "0700.HK", "source_type": "synthetic", "published_at": when.isoformat(),
            "valid_to": None, "text": text, "extracted_text": text, "content_digest": digest(text)}


def proposal(eid="e1", text="Revenue was 10 units.", when=WHEN, **changes):
    values = dict(ticker="0700.HK", claim_type="fundamental", subject="company", predicate="reported", metric="revenue",
                  period="2024FY", value="10", text=text, polarity="neutral", evidence_ids=[eid], quotes={eid:text}, available_at=when)
    values.update(changes)
    return ClaimProposal(**values)


def review(index=0, eid="e1", supported=True):
    return Verification(proposal_index=index, supported=supported, evidence_ids=[eid], reason="Test fixture")


def test_revision_preserves_historical_snapshot_and_is_idempotent():
    ledger = GraphLedger("0700.HK")
    old = ledger.commit([proposal()], [review()], [source()], WHEN)["accepted"][0]
    later = WHEN + timedelta(days=2)
    revised = proposal(eid="e2", text="Revenue revised to 11 units.", when=later, value="11", revision_of=old)
    ledger.commit([revised], [review(eid="e2")], [source("e2", revised.text, later)], later)
    assert [c["entity_id"] for c in ledger.freeze(WHEN)["claims"]] == [old]
    assert ledger.freeze(WHEN)["claims"][0]["is_active"] is True
    assert ledger.freeze(WHEN)["claims"][0]["valid_to"] is None
    assert "superseded_by" not in ledger.freeze(WHEN)["claims"][0]
    assert ledger.freeze(later)["claims"][0]["version"] == 2
    assert ledger.freeze(later)["conflicts"] == []
    assert ledger.state["claims"][0]["valid_to"] == later.isoformat()
    before = digest(ledger.state)
    ledger.commit([revised], [review(eid="e2")], [source("e2", revised.text, later)], later)
    assert digest(ledger.state) == before
    assert any(e["type"] == "SUPERSEDES" for e in ledger.state["relations"])


def test_conflict_requires_matching_metric_period_and_subject():
    ledger = GraphLedger("0700.HK")
    ps = [proposal(), proposal(value="11", polarity="contradicts"), proposal(metric="earnings", value="11"),
          proposal(period="2023FY", value="11"), proposal(subject="other company", value="11")]
    ledger.commit(ps, [review(i) for i in range(len(ps))], [source()], WHEN)
    assert len(ledger.freeze(WHEN)["conflicts"]) == 1


@pytest.mark.parametrize("change", [{"ticker":"0005.HK"}, {"available_at":WHEN+timedelta(days=1)},
    {"quotes":{"e1":"invented number"}}, {"evidence_ids":["bad"],"quotes":{"bad":"x"}},
    {"valid_to":WHEN}, {"revision_of":"nonexistent"}])
def test_invalid_proposals_rejected_without_claim(change):
    ledger = GraphLedger("0700.HK")
    p = proposal(**change)
    delta = ledger.commit([p], [Verification(proposal_index=0,supported=True,evidence_ids=p.evidence_ids,reason="fixture")], [source()], WHEN)
    assert len(delta["rejected"]) == 1
    assert not ledger.state["claims"]


def test_verifier_rejection_and_missing_review():
    ledger = GraphLedger("0700.HK")
    assert ledger.commit([proposal()], [review(supported=False)], [source()], WHEN)["rejected"]
    with pytest.raises(ValueError, match="exactly one"):
        ledger.commit([proposal()], [], [source()], WHEN)


def test_atomic_failed_store_does_not_advance_memory():
    class FailedStore:
        def commit_research_graph(self,*args): raise RuntimeError("transaction failed")
    ledger = GraphLedger("0700.HK")
    before = digest(ledger.state)
    with pytest.raises(RuntimeError):
        ledger.commit([proposal()], [review()], [source()], WHEN,store=FailedStore(),namespace="test")
    assert digest(ledger.state) == before


def test_evidence_revision_requires_distinct_identity():
    ledger = GraphLedger("0700.HK")
    ledger.commit([],[],[source()],WHEN)
    with pytest.raises(ValueError, match="new identity"):
        ledger.commit([],[],[source(text="Changed contents")],WHEN)


def test_unknown_future_expired_and_other_ticker_excluded():
    raw = {"evidences":[source(), {**source("unknown"),"published_at":None}, source("future",when=WHEN+timedelta(days=1)),
                         {**source("expired"),"valid_to":WHEN.isoformat()}, {**source("other"),"ticker":"0005.HK"}]}
    corpus, coverage = evidence_corpus(raw,"0700.HK",WHEN)
    assert [r["evidence_id"] for r in corpus] == ["e1"]
    assert coverage["excluded_unavailable"] == 4


def test_expiry_does_not_destroy_history():
    ledger = GraphLedger("0700.HK")
    expiry = WHEN + timedelta(days=1)
    ledger.commit([proposal(valid_to=expiry)], [review()], [source()], WHEN)
    ledger.commit([],[],[],expiry)
    assert not ledger.freeze(expiry)["claims"]
    assert len(ledger.freeze(WHEN)["claims"]) == 1
    ledger.commit([],[],[],expiry)
    assert sum(e["type"] == "expired" for e in ledger.state["events"]) == 1


class Judge:
    def __init__(self, bad=False): self.calls=0; self.bad=bad
    def generate_json(self, system, user):
        self.calls += 1
        inputs=json.loads(user)
        evidence={r["evidence_id"]:r["text"] for r in inputs["evidence"]}
        return {"units":[{"text":"Revenue was 10 units.","kind":"factual","verdict":"supported", "quotes": {next(iter(evidence)):"invented" if self.bad else next(iter(evidence.values()))}}] if evidence else [],
                "citations":[{"evidence_id":eid,"relevant":eid in evidence,"quote":evidence.get(eid,"")} for eid in inputs["cited_ids"]]}


def decision():
    return {"action":"buy","decision_reason":"Revenue supports outlook", "evidence_refs":["e1"],
            "agent_reports":[{"role":"fundamental","summary":"Revenue was 10 units."}]}


def test_semantic_cache_and_quote_integrity(tmp_path):
    client=Judge(); evaluator=SemanticEvaluator(client,tmp_path)
    first=evaluator.evaluate(decision(),[source()],key_evidence_ids=["e1"])
    assert first["faithfulness_score"] == first["semantic_citation_precision"] == first["key_evidence_coverage"] == 1
    assert not first["cache_hit"]
    assert evaluator.evaluate(decision(),[source()],key_evidence_ids=["e1"])["cache_hit"]
    assert client.calls == 1
    path=tmp_path/(first["cache_key"]+".json")
    saved=json.loads(path.read_text());saved["response"]["units"][0]["text"]="tampered"
    write_json(path,saved)
    with pytest.raises(ValueError,match="response digest"):
        evaluator.evaluate(decision(),[source()],key_evidence_ids=["e1"])


def test_invalid_judge_not_cached_and_missing_judge_not_fallback(tmp_path):
    with pytest.raises(ValueError,match="quote"):
        SemanticEvaluator(Judge(True),tmp_path).evaluate(decision(),[source()])
    assert not list(tmp_path.glob("*.json"))
    with pytest.raises(RuntimeError,match="judge is required"):
        SemanticEvaluator(None,tmp_path).evaluate(decision(),[source()])


def test_semantic_false_support_and_unmeasured_coverage(tmp_path):
    response=JudgeResponse.model_validate({"units":[{"text":"Revenue was 12", "kind":"factual", "verdict":"contradicted", "quotes":{"e1":"Revenue was 10 units."}},
                        {"text":"A forecast", "kind":"inference", "verdict":"not_found", "quotes":{}}],
                        "citations":[{"evidence_id":"e1","relevant":False}]})
    result=score_judgment(response,[source()],["e1"])
    assert result["faithfulness_score"] == result["semantic_citation_precision"] == 0
    assert result["factual_hallucination_rate"] == result["hallucination_rate"] == 1
    assert result["key_evidence_coverage"] is None
    assert result["coverage_status"] == "not_annotated"


def test_static_missing_keys_are_missed_not_invalid(tmp_path):
    result=SemanticEvaluator(Judge(),tmp_path).evaluate(decision(),[source()],key_evidence_ids=["e2"],reference_corpus=[source(),source("e2")])
    assert result["key_evidence_coverage"] == 0
    assert result["n_key_evidence_outside_input"] == 1
    with pytest.raises(ValueError,match="reference corpus"):
        SemanticEvaluator(Judge(),tmp_path).evaluate(decision(),[source()],key_evidence_ids=["fake"])


def test_independent_annotation_agreement():
    result=annotation_agreement([{"human":"supported","judge":"supported"},{"human":"not_found","judge":"supported"}])
    assert result["n_units"] == 2 and result["agreement"] == .5 and result["cohen_kappa"] == 0
    with pytest.raises(ValueError): annotation_agreement([])


class Constructor:
    def __init__(self): self.calls=0
    def generate_json(self,system,user):
        self.calls+=1; data=json.loads(user); e=data["evidence"][0]
        payload={"proposals":[proposal(eid=e["evidence_id"],text=e["text"],when=pd.Timestamp(e["published_at"]).to_pydatetime(),period=e["published_at"]).model_dump(mode="json")]}
        record_call({"parsed_response":payload});return payload


class Verifier:
    def __init__(self):self.calls=0
    def generate_json(self,system,user):
        self.calls+=1; data=json.loads(user)
        payload={"reviews":[Verification(proposal_index=i,supported=True,evidence_ids=p["evidence_ids"],reason="Synthetic verification").model_dump() for i,p in enumerate(data["proposals"])]}
        record_call({"parsed_response":payload});return payload


class Analyst:
    def __init__(self):self.budgets=[]
    def generate_json_with_budget(self,system,user,max_tokens):
        self.budgets.append(max_tokens); data=json.loads(user)
        view=data["input"]
        if view.startswith("{"):
            corpus=json.loads(view)["evidences"]
            eid=corpus[0]["evidence_id"] if corpus else None
        else:
            import re
            ids=re.findall(r'evidence_id="([^"\n]+)"',view);eid=ids[0] if ids else None
        report={"stance":"bullish" if eid else "uncertain","confidence":.8 if eid else .1,"score":.5 if eid else 0,
                "summary":"Revenue was 10 units." if eid else "No evidence", "evidence_refs":[eid] if eid else [],"claim_refs":[],"factors":[]}
        payload={"reports":{role:deepcopy(report) for role in ("technical","news","fundamental","risk")}} if "Return {reports:" in system else report
        record_call({"parsed_response":payload});return payload


class Query:
    def __init__(self):self.drift=False
    def load_subgraph(self,ticker,when):
        return {"evidences":[source(eid=when.date().isoformat(),text="Revenue was 11 units." if self.drift else "Revenue was 10 units.",when=when)]}


def market(ticker):
    return pd.DataFrame({"date":pd.date_range("2025-01-01",periods=12).strftime("%Y-%m-%d"),"open":range(100,112),"close":range(101,113)})


def run_fixture(tmp_path, *, cfg=None, query=None, judge=None):
    clients=(Analyst(),Constructor(),Verifier(),judge or Judge())
    cfg=cfg or ResearchConfig(output_dir=tmp_path,run_id="test")
    df=run_research(["0700.HK"],["2025-01-02","2025-01-03"],config=cfg,query_client=query or Query(),ohlcv_loader=market,
                    analyst_client=clients[0],proposer_client=clients[1],verifier_client=clients[2],judge_client=clients[3])
    return df,clients,cfg


def test_five_controlled_variants_end_to_end_resume_and_replay(tmp_path):
    df,clients,cfg=run_fixture(tmp_path)
    assert len(df)==10 and set(df.system)==set(VARIANTS)
    assert set(df.execution_status)=={"success"}
    for date,g in df.groupby("trade_date"):
        assert g[g.system!="static_graph"].corpus_digest.nunique()==1
    static=df[df.system=="static_graph"]
    assert static.corpus_digest.nunique()==1
    assert df.initial_state_digest.nunique() == 1
    assert clients[0].budgets.count(2048)==2
    assert clients[0].budgets.count(512)==32
    path=Path(df.attrs["run_dir"])
    updates = [json.loads(p.read_text()) for p in (path / "updates").glob("*.json")]
    initial, first, second = sorted(updates, key=lambda row: row["observed"]["cutoff"])
    assert initial["state_digest"] == df.initial_state_digest.iloc[0] == first["observed"]["before_digest"]
    assert first["state_digest"] == second["observed"]["before_digest"]
    for task in (path/"tasks").glob("*.json"):
        r=json.loads(task.read_text());assert replay_record(r)["decision"]["action"]==r["decision"]["action"]
    resumed,newclients,_=run_fixture(tmp_path,cfg=replace(cfg,resume=True))
    assert len(resumed)==10 and not newclients[0].budgets and newclients[1].calls==0 and newclients[2].calls==0 and newclients[3].calls==0
    report=json.loads((path/"research_report.json").read_text())
    assert all(c["n_pairs"]==2 for c in report["comparisons"])
    assert all(c["n_support_pairs"] == 2 and c["mean_net_return_delta"] == 0 for c in report["comparisons"])
    assert all(c["same_corpus"] for c in report["comparisons"] if c["ablation"]!="static_graph")


def test_seed_cutoff_equal_to_first_decision_uses_one_update(tmp_path):
    cfg = ResearchConfig(output_dir=tmp_path, run_id="equal", static_cutoff="2025-01-02")
    df, clients, _ = run_fixture(tmp_path, cfg=cfg)
    assert len(df) == 10
    assert clients[1].calls == clients[2].calls == 2
    assert df.loc[df.trade_date == "2025-01-02", "snapshot_digest"].nunique() == 1


def test_eligible_indicators_are_visible_in_both_representations():
    record = {"ticker": "0700.HK", "as_of_date": WHEN.isoformat()}
    corpus, coverage = evidence_corpus({
        "signals": [
            {**record, "entity_id": "tech", "name": "MA20", "ma20": 24.2},
            {**record, "entity_id": "future", "name": "MA50", "as_of_date": (WHEN + timedelta(days=1)).isoformat()},
        ],
        "fundamentals": [{**record, "entity_id": "fund", "title": "Results", "metric": "revenue",
                          "numeric_value": 17.3, "period": "FY2024"}],
    }, "0700.HK", WHEN)
    assert coverage["excluded_unavailable"] == 1
    ledger = GraphLedger("0700.HK")
    ledger.commit([], [], corpus, WHEN)
    snapshot = ledger.freeze(WHEN)
    for representation in ("graph", "text"):
        view = frozen_view(snapshot, representation)
        assert all(value in view for value in ("MA20", "24.2", "FY2024", "17.3"))
        assert "MA50" not in view


def test_evidence_perturbation_records_paired_views_without_modifying_source(tmp_path):
    df, _, _ = run_fixture(tmp_path)
    run_dir = Path(df.attrs["run_dir"])
    task_path = run_dir / "tasks" / (digest(["0700.HK", "2025-01-02", "full_graph"]) + ".json")
    original = task_path.read_bytes()
    snapshot = json.loads(original)["decision_snapshot"]
    key = "2025-01-02"
    edge = next(row for row in snapshot["relations"] if row["end_id"] == key)
    case = {"ticker": "0700.HK", "trade_date": "2025-01-02", "key_evidence_id": key,
            "control_evidence_id": "2025-01-01", "critical_relation": {
                field: edge[field] for field in ("start_id", "end_id", "type")
            }}
    views = _views(snapshot, case)
    assert all(key not in claim["evidence_ids"] for claim in views["without_key_evidence"]["claims"])
    assert all(rel["end_id"] != key for rel in views["without_key_evidence"]["relations"])
    assert edge not in views["without_critical_relation"]["relations"]
    assert views["without_critical_relation"]["evidences"] == snapshot["evidences"]
    analyst = Analyst()
    rows = run_evidence_perturbation(run_dir, tmp_path / "diagnostic", [case], analyst_client=analyst)
    assert len(rows) == 4 and all(row["status"] == "success" for row in rows)
    assert len({row["snapshot_digest"] for row in rows}) == 4
    assert analyst.budgets == [512] * 16
    assert task_path.read_bytes() == original


def test_multihop_paths_reach_evidence_without_query_overlap():
    ledger = GraphLedger("0700.HK")
    statements = {
        "anchor": "Revenue growth accelerated.",
        "disagreement": "Turnover was eight units.",
        "additional": "Warehouse stocks declined.",
    }
    corpus = [source(key, value) for key, value in statements.items()]
    claims = [
        proposal("anchor", statements["anchor"]),
        proposal("disagreement", statements["disagreement"], value="8"),
        proposal("additional", statements["additional"], metric="inventory",
                 evidence_ids=["anchor", "additional"], quotes={
                     "anchor": statements["anchor"], "additional": statements["additional"],
                 }),
    ]
    reviews = [review(0, "anchor"), review(1, "disagreement"), Verification(
        proposal_index=2, supported=True, evidence_ids=["anchor", "additional"], reason="Synthetic test",
    )]
    ledger.commit(claims, reviews, corpus, WHEN)
    snapshot = ledger.freeze(WHEN)
    before = digest(snapshot)
    results = retrieve_research_evidence(snapshot, "0700.HK", WHEN, "revenue growth", top_k=3, seed_limit=1)
    assert results["text_lexical"]["evidence_ids"] == results["graph_one_hop"]["evidence_ids"] == ["anchor"]
    multihop = results["graph_multihop"]
    assert set(multihop["new_beyond_direct_support"]) == {"disagreement", "additional"}
    assert {p["rule"]: p["hops"] for p in multihop["paths"]} == {
        "conflicting_support": 2, "direct_support": 1, "shared_source_support": 3,
    }
    for path in multihop["paths"]:
        assert len(set(path["nodes"])) == len(path["nodes"])
        assert path["edges"][-1]["quote"] in statements[path["evidence_id"]]
    for rule, remaining in (("conflicting_support", {"anchor", "additional"}),
                            ("shared_source_support", {"anchor", "disagreement"})):
        result = retrieve_research_evidence(
            snapshot, "0700.HK", WHEN, "revenue growth", top_k=3, seed_limit=1, disabled_rule=rule
        )
        assert set(result["graph_multihop"]["evidence_ids"]) == remaining
        assert result["graph_one_hop"] == results["graph_one_hop"]
    assert digest(snapshot) == before

    snapshot["relations"][0]["effective_at"] = (WHEN + timedelta(days=1)).isoformat()
    with pytest.raises(ValueError, match="unavailable"):
        retrieve_research_evidence(snapshot, "0700.HK", WHEN, "revenue")


def test_multihop_saved_snapshot_reports_annotation_and_decision_pairs(tmp_path):
    df, _, _ = run_fixture(tmp_path)
    source_run = Path(df.attrs["run_dir"])
    tasks = source_run / "tasks"
    original = {p.name: p.read_bytes() for p in tasks.glob("*.json")}
    cases = [
        {"ticker": "0700.HK", "trade_date": "2025-01-02", "query": "revenue",
         "key_evidence_ids": ["2025-01-02"]},
        {"ticker": "0700.HK", "trade_date": "2025-01-03", "query": "unmatched"},
    ]
    output = tmp_path / "diagnostic"
    analyst = Analyst()
    rows = run_multihop(source_run, output, cases, top_k=2, analyst_client=analyst)
    assert len(rows) == 6 and len(analyst.budgets) == 24
    assert all(row["decision_status"] == "success" for row in rows)
    assert all(row["key_recall"] == 1 for row in rows if row["trade_date"] == "2025-01-02")
    assert all(row["key_recall"] is None and row["retrieved_count"] == 0
               for row in rows if row["trade_date"] == "2025-01-03")
    summary = json.loads((output / "retrieval_summary.json").read_text(encoding="utf-8"))
    assert all(c["n_annotated_pairs"] == 1 and c["n_successful_settled_pairs"] == 2
               for c in summary["comparisons"])
    ablated = run_multihop(source_run, tmp_path / "ablations", cases, ablate_path_rules=True)
    assert len(ablated) == 10
    assert {row["variant"] for row in ablated} >= {
        "without_conflicting_support", "without_shared_source_support",
    }
    assert original == {p.name: p.read_bytes() for p in tasks.glob("*.json")}


def test_resume_source_drift_rejected(tmp_path):
    df,_,cfg=run_fixture(tmp_path);query=Query();query.drift=True
    tasks=Path(df.attrs["run_dir"])/"tasks";before={p.name:p.read_bytes() for p in tasks.glob("*.json")}
    with pytest.raises(ValueError,match="snapshot changed"):
        run_fixture(tmp_path,cfg=replace(cfg,resume=True),query=query)
    assert before=={p.name:p.read_bytes() for p in tasks.glob("*.json")}


def test_judge_failure_is_recorded_not_perfect_score(tmp_path):
    df,_,_=run_fixture(tmp_path,judge=Judge(True))
    assert df.empty
    path=Path(df.attrs["run_dir"])
    summary=json.loads((path/"run_summary.json").read_text())
    assert summary["status_counts"]["failed"]==10
    records=[json.loads(p.read_text()) for p in (path/"tasks").glob("*.json")]
    assert all(r["issues"][0]["stage"]=="semantic_grounding" and r["row"] is None for r in records)


def test_export_preflight_no_partial_files_on_corpus_mismatch(tmp_path):
    df,_,_=run_fixture(tmp_path/"run")
    df.loc[df.system=="text_same_evidence","corpus_digest"]="different"
    output=tmp_path/"bad"
    with pytest.raises(ValueError,match="corpus mismatch"):
        export_research(df,output,updates=pd.DataFrame())
    assert not output.exists()


def test_synthetic_risk_benchmark_and_latency(tmp_path):
    from src.eval.risk_benchmark import run_risk_benchmark
    df=run_risk_benchmark(tmp_path,repetitions=2)
    assert len(df)==16
    gated=df[df.policy=="conservative_v1"].groupby("scenario").action.first().to_dict()
    assert gated=={"empty_evidence":"abstain","fresh_conflicting":"hold","stale_evidence":"abstain","fresh_consistent":"buy"}
    assert set(df[df.policy=="observational"].action)=={"buy"}
    assert pd.read_csv(tmp_path/"synthetic_memory_update_latency.csv").memory_commit_and_freeze_latency.ge(0).all()


def test_response_budget_must_match_across_architectures():
    with pytest.raises(ValueError):ResearchConfig(response_token_budget=2049)


def test_source_urls_are_not_serialized(tmp_path):
    corpus,_=evidence_corpus({"evidences":[source(text="Revenue was 10 units. https://example.invalid?api_key=hidden")]},"0700.HK",WHEN)
    assert "https://" not in corpus[0]["text"] and "hidden" not in corpus[0]["text"]


def test_research_neo4j_commit_constraints_atomic_callback_and_stale_state():
    from src.kg.store_neo4j import Neo4jKGStore
    class Result:
        def __init__(self,value=None):self.value=value
        def single(self):return self.value
        def consume(self):return None
    class Driver:
        def __init__(self):self.calls=[];self.state_digest=None;self.write_calls=0
        def session(self,**kwargs):return self
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def run(self,cypher,**params):
            self.calls.append((cypher,params))
            if "RETURN s.state_digest" in cypher:
                return Result({"state_digest":self.state_digest or params["before"]})
            if "SET s.state_digest = $after" in cypher:self.state_digest=params["after"]
            return Result()
        def execute_write(self,callback):self.write_calls+=1;callback(self)
    store=object.__new__(Neo4jKGStore);store._driver=Driver();store._database="test"
    ledger=GraphLedger("0700.HK");before=digest(ledger.state)
    ledger.commit([proposal()],[review()],[source()],WHEN,store=store,namespace="test_run")
    assert store._driver.write_calls==1
    constraints=[q for q,p in store._driver.calls if "CREATE CONSTRAINT" in q]
    assert len(constraints)==3 and all("Research" in q and "UNIQUE" in q for q in constraints)
    statements=[q for q,p in store._driver.calls if "MERGE" in q]
    assert all("Research" in q or "RESEARCH_LINK" in q for q in statements)
    assert store._driver.state_digest==digest(ledger.state)
    before_calls=len(store._driver.calls)
    store.commit_research_graph("test_run","0700.HK",before,ledger.state)
    assert len(store._driver.calls)==before_calls+1
    store._driver.state_digest="other_writer"
    with pytest.raises(ValueError,match="concurrent overwrite"):
        store.commit_research_graph("test_run","0700.HK",before,ledger.state)


def test_failed_semantic_attempts_retained_on_successful_retry(tmp_path):
    df,_,cfg=run_fixture(tmp_path,judge=Judge(True))
    path=Path(df.attrs["run_dir"])
    before={p.name:p.read_bytes() for p in (path/"tasks").glob("*.json")}
    recovered,_,_=run_fixture(tmp_path,cfg=replace(cfg,resume=True),judge=Judge())
    assert len(recovered)==10
    archived=list((path/"attempts").glob("*.json"))
    assert len(archived)==10
    assert all(p.read_bytes() in before.values() for p in archived)
    assert all(json.loads(p.read_text())["status"]=="failed" for p in archived)


def test_graph_failure_has_captured_diagnostics_and_no_fallback(tmp_path):
    class BrokenConstructor(Constructor):
        def generate_json(self,system,user):
            record_call({"error_type":"ValueError","parsed_response":{"proposals":"invalid"}})
            raise ValueError("malformed structured output")
    df=run_research(["0700.HK"],["2025-01-02"],config=ResearchConfig(output_dir=tmp_path,run_id="broken"),
                    query_client=Query(),ohlcv_loader=market,analyst_client=Analyst(),proposer_client=BrokenConstructor(),
                    verifier_client=Verifier(),judge_client=Judge())
    assert df.empty
    path=Path(df.attrs["run_dir"])
    records=[json.loads(p.read_text()) for p in (path/"tasks").glob("*.json")]
    assert len(records)==5 and all(r["status"]=="failed" and r["calls"] for r in records)
    assert len(list((path/"update_attempts").glob("*.json")))==1


def test_saved_source_rejects_missing_scope_and_invalid_digest(tmp_path):
    from src.eval.research_sources import SavedRunSource
    frame=market("0700.HK").to_dict("records")
    inputs={"snapshot":{"evidences":[source()]},"ohlcv":frame,"effective_cutoff":WHEN.isoformat()}
    record={"status":"success","system":"kg_dynamic","ticker":"0700.HK","trade_date":"2025-01-02", "inputs":inputs,
            "input_digest":digest(inputs),"decision_snapshot":inputs["snapshot"],"decision_snapshot_digest":digest(inputs["snapshot"])}
    write_json(tmp_path/"tasks/a.json",record)
    adapter=SavedRunSource(tmp_path)
    assert len(adapter.load_ohlcv("0700.HK"))==12
    assert adapter.load_subgraph("0700.HK",WHEN)["evidences"][0]["evidence_id"]=="e1"
    with pytest.raises(ValueError,match="requested ticker/cutoff"):
        adapter.load_subgraph("0005.HK",WHEN)
    record["input_digest"]="bad";write_json(tmp_path/"tasks/a.json",record)
    with pytest.raises(ValueError,match="integrity"):
        SavedRunSource(tmp_path)


def test_historical_review_keeps_originals_and_labels_semantics(tmp_path):
    from src.eval.records import RunRecorder
    from src.eval.research_reporting import historical_review
    from src.agents.models import AgentReport
    from src.agents.pm_deterministic import portfolio_manager_decide
    from src.eval.protocol import backtest_details
    from src.sim.backtest import BacktestConfig
    recorder=RunRecorder(tmp_path/"original", {"protocol_version":"pit_v1", "tickers":["0700.HK"],
                                "trade_dates":["2025-01-02","2025-01-03","2025-01-04"]},run_id="old")
    snapshot={"company":[],"evidences":[source(text="Revenue was 10 units. https://example.invalid?token=hidden")],"claims":[],"signals":[],"news":[],"risks":[],"fundamentals":[]}
    records=[]
    for date in ("2025-01-02","2025-01-03","2025-01-04"):
        when=decision_cutoff(date)
        for system in ("kg_dynamic","static_kg","evidence_no_kg","no_kg_no_evidence"):
            reports=[AgentReport(role,"bullish",.8,.5,"Revenue was 10 units.",["e1"],[],[]) for role in ("technical","news","fundamental","risk")]
            d=portfolio_manager_decide("0700.HK",when,reports,snapshot)
            bt=backtest_details(market("0700.HK"),date,d["action"],BacktestConfig(),protocol_version="pit_v1")
            row={**d,**bt,"system":system,"execution_status":"success"}
            inputs={"protocol_version":"pit_v1","decision_time":when.isoformat(),"snapshot":snapshot,
                    "ohlcv":market("0700.HK").to_dict("records"),"backtest_config":{"holding_days":5,"transaction_cost_bp":5.0}}
            records.append(recorder.save("0700.HK",date,system,inputs,status="success",row=row,decision=d,backtest=bt,decision_snapshot=snapshot))
    recorder.finish(records)
    before={p:p.read_bytes() for p in recorder.path.rglob("*.json")}
    output=tmp_path/"review"
    result=historical_review(recorder.path,output)
    assert result["n_tasks"]==12 and result["case_count"]==3 and not result["replay_errors"]
    assert result["source_tasks_unchanged"]
    assert all(p.read_bytes()==v for p,v in before.items())
    cases=(output/"case_studies.json").read_text()
    assert "https://" not in cases and "hidden" not in cases
    assert "same_date_comparison" in cases and "does not establish" in cases
    with pytest.raises(ValueError,match="separate output"):
        historical_review(recorder.path,recorder.path/"review")
