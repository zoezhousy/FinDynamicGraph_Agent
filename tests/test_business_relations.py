"""Synthetic evidence, temporal lifecycle and path-proof regression tests."""
from copy import deepcopy
from datetime import timedelta
import json

import pytest
from pydantic import ValidationError

from src.eval.protocol import decision_cutoff
from src.eval.thesis_sources import LocalEvidence
from src.kg.business_relations import BusinessProposal, BusinessUpdater, METRIC
from src.kg.collaboration import GraphLedger, Verification
from src.kg.thesis_paths import freeze_graph, EvidenceGraph
from src.kg.trace_paths import mine_trace_rules, explore_trace, validate_context


TEXT = ("Alpha partnered with Beta. Alpha invested in Beta. Beta sells Widget. "
        "Alpha completed acquisition of Beta. Alpha completed divestment of part of its stake in Beta.")


def source(day="2024-01-02", text=TEXT):
    raw = LocalEvidence([{"ticker": "0700.HK", "source_type": "hkexnews", "source": "fixture",
                         "document_id": day, "version": "1", "language": "en",
                         "published_at": day + "T09:00:00+08:00", "text": text}])
    return raw.snapshot("0700.HK", decision_cutoff(day))


def proposal(raw, relation="PARTNERED", *, head=None, tail=None, quote=None, **changes):
    evidence = raw["evidences"][0]
    payload = {"ticker": "0700.HK", "relation": relation,
               "head": head or {"kind": "Company", "name": "Alpha", "ticker": "0700.HK"},
               "tail": tail or {"kind": "Company", "name": "Beta"},
               "evidence_ids": [evidence["evidence_id"]],
               "quotes": {evidence["evidence_id"]: quote or TEXT.split(".")[0] + "."},
               "available_at": evidence["published_at"], **changes}
    return BusinessProposal.model_validate(payload)


def commit(ledger, proposals, raw, day="2024-01-02", supported=True):
    from src.kg.collaboration import evidence_corpus
    corpus, _ = evidence_corpus(raw, "0700.HK", decision_cutoff(day), strict=True)
    return ledger.commit([row.claim() for row in proposals], [Verification(
        proposal_index=i, supported=supported, evidence_ids=row.evidence_ids, reason="fixture independent verification")
        for i, row in enumerate(proposals)], corpus, decision_cutoff(day))


@pytest.mark.parametrize("relation", ["PARTNERED", "INVESTED_IN", "SELLS", "ACQUIRED", "DIVESTED"])
def test_typed_relation_projection(relation):
    raw = source()
    tail = {"kind": "Product", "name": "Widget"} if relation == "SELLS" else {"kind": "Company", "name": "Beta"}
    head = {"kind": "Company", "name": "Beta"} if relation == "SELLS" else None
    quote = {"PARTNERED": "Alpha partnered with Beta.", "INVESTED_IN": "Alpha invested in Beta.",
             "SELLS": "Beta sells Widget.", "ACQUIRED": "Alpha completed acquisition of Beta.",
             "DIVESTED": "Alpha completed divestment of part of its stake in Beta."}[relation]
    extra = {"transaction_status": "completed"} if relation in {"ACQUIRED", "DIVESTED"} else {}
    p = proposal(raw, relation, head=head, tail=tail, quote=quote, **extra)
    ledger = GraphLedger("0700.HK", strict_cutoff=True)
    assert len(commit(ledger, [p], raw)["accepted"]) == 1
    graph = freeze_graph(ledger, raw, decision_cutoff("2024-01-02"), limit=8)
    edge, = [row for row in graph["relations"] if row["type"] == relation]
    assert edge["quotes"] == p.quotes
    assert edge["start_id"] == p.claim().subject and edge["end_id"] == p.claim().value
    assert any(row["type"] == tail["kind"] for row in graph["business_entities"])


@pytest.mark.parametrize("relation", ["ACQUIRED", "DIVESTED"])
@pytest.mark.parametrize("status", [None, "announced", "pending", "cancelled"])
def test_transaction_must_be_explicitly_completed(relation, status):
    with pytest.raises(ValidationError, match="completed"):
        proposal(source(), relation, transaction_status=status)


@pytest.mark.parametrize("relation", ["ACQUIRED", "DIVESTED"])
def test_completed_business_path_mining_and_inverse(relation):
    raw = source()
    quote = "Alpha completed acquisition of Beta." if relation == "ACQUIRED" else "Alpha completed divestment of part of its stake in Beta."
    p = proposal(raw, relation, quote=quote, transaction_status="completed")
    ledger = GraphLedger("0700.HK", strict_cutoff=True)
    commit(ledger, [p], raw)
    cutoff = decision_cutoff("2024-01-02")
    graph = freeze_graph(ledger, raw, cutoff, limit=8)
    edge = next(row for row in graph["relations"] if row["type"] == relation)
    assert (edge["start_id"], edge["end_id"]) == (p.claim().subject, p.claim().value)
    assert any(kind == "^" + relation for kind, _, _ in EvidenceGraph(graph, cutoff).edges[edge["end_id"]])
    rules, _ = mine_trace_rules([(graph, cutoff + timedelta(days=i), "UP") for i in range(5)])
    rules = [rule for rule in rules if rule["body"] == [relation, "EXTRACTED_FROM"]]
    assert rules
    context, _ = explore_trace(graph, cutoff, rules)
    validate_context(graph, cutoff, context)
    assert quote in context["evidences"][0]["text"]


def test_partial_divestment_keeps_investment_and_legacy_payloads():
    raw = source()
    ledger = GraphLedger("0700.HK", strict_cutoff=True)
    investment = proposal(raw, "INVESTED_IN", quote="Alpha invested in Beta.")
    assert "transaction_status" not in investment.model_dump()
    assert "transaction_status" not in json.loads(investment.claim().text)
    disposal = proposal(raw, "DIVESTED", transaction_status="completed",
                        quote="Alpha completed divestment of part of its stake in Beta.")
    assert len(commit(ledger, [investment, disposal], raw)["accepted"]) == 2
    graph = freeze_graph(ledger, raw, decision_cutoff("2024-01-02"), limit=8)
    assert {"INVESTED_IN", "DIVESTED"} <= {row["type"] for row in graph["relations"]}


@pytest.mark.parametrize("failure", ["quote", "name", "verifier", "future", "source_type", "mapped_ticker"])
def test_unverified_relations_never_create_business_edges(failure):
    raw = source()
    changes = {}
    if failure == "quote": changes["quote"] = "Alpha partnered with Beta. Invented."
    if failure == "name": changes["tail"] = {"kind": "Company", "name": "Gamma"}
    if failure == "future": changes["valid_from"] = "2025-01-01T09:00:00+08:00"
    if failure == "mapped_ticker": changes["tail"] = {"kind": "Company", "name": "Beta", "ticker": "0005.HK"}
    if failure == "source_type": raw["evidences"][0]["source_type"] = "technical"
    p = proposal(raw, **changes)
    ledger = GraphLedger("0700.HK", strict_cutoff=True)
    assert commit(ledger, [p], raw, supported=failure != "verifier")["rejected"]
    graph = freeze_graph(ledger, raw, decision_cutoff("2024-01-02"), limit=8)
    assert not graph.get("business_entities")


def test_wrong_endpoint_type_rejected():
    with pytest.raises(ValidationError):
        proposal(source(), "SELLS")


def test_source_owner_reaches_typed_claim_without_merging_subsidiary():
    raw = source()
    p = proposal(raw, head={"kind": "Company", "name": "Alpha"})
    ledger = GraphLedger("0700.HK", strict_cutoff=True)
    commit(ledger, [p], raw)
    cutoff = decision_cutoff("2024-01-02")
    snapshot = freeze_graph(ledger, raw, cutoff, limit=8)
    edge = next(row for row in snapshot["relations"] if row["type"] == "PARTNERED")
    assert "company:0700.HK" not in (edge["start_id"], edge["end_id"])
    rules, _ = mine_trace_rules([(snapshot, cutoff + timedelta(days=i), "UP") for i in range(5)])
    typed = [rule for rule in rules if rule["body"] == ["HAS_VERIFIED_PARTNERED_CLAIM", "SUPPORTED_BY"]]
    assert typed
    context, _ = explore_trace(snapshot, cutoff, typed)
    validate_context(snapshot, cutoff, context)
    assert context["paths"]
    forged = deepcopy(snapshot)
    provenance = next(row for row in forged["relations"] if row["type"] == "HAS_VERIFIED_PARTNERED_CLAIM")
    provenance["type"] = "HAS_VERIFIED_ACQUIRED_CLAIM"
    assert not any(kind == "HAS_VERIFIED_ACQUIRED_CLAIM"
                   for kind, _, _ in EvidenceGraph(forged, cutoff).edges["company:0700.HK"])


def test_multiple_products_coexist_and_commit_idempotent():
    raw = source(text="Alpha sells Widget and Gadget.")
    a = proposal(raw, "SELLS", tail={"kind": "Product", "name": "Widget"}, quote="Alpha sells Widget and Gadget.")
    b = proposal(raw, "SELLS", tail={"kind": "Product", "name": "Gadget"}, quote="Alpha sells Widget and Gadget.")
    ledger = GraphLedger("0700.HK", strict_cutoff=True)
    commit(ledger, [a, b], raw)
    commit(ledger, [a, b], raw)
    assert len(ledger.state["claims"]) == 2
    assert not [edge for edge in ledger.state["relations"] if edge["type"] == "CONFLICTS_WITH"]


def test_expiry_and_revision_preserve_historical_projection():
    raw = source()
    ledger = GraphLedger("0700.HK", strict_cutoff=True)
    a = proposal(raw)
    old_id = commit(ledger, [a], raw)["accepted"][0]
    original = freeze_graph(ledger, raw, decision_cutoff("2024-01-02"), limit=8)
    later = source("2024-01-03")
    b = proposal(later, revision_of=old_id, valid_to="2024-01-05T09:00:00+08:00")
    commit(ledger, [b], later, day="2024-01-03")
    assert freeze_graph(ledger, raw, decision_cutoff("2024-01-02"), limit=8) == original
    assert not freeze_graph(ledger, later, decision_cutoff("2024-01-06"), limit=8).get("business_entities")


def test_business_paths_keep_every_edge_proof():
    raw = source()
    ledger = GraphLedger("0700.HK", strict_cutoff=True)
    partner = proposal(raw)
    sells = proposal(raw, "SELLS", head={"kind": "Company", "name": "Beta"},
                     tail={"kind": "Product", "name": "Widget"}, quote="Beta sells Widget.")
    commit(ledger, [partner, sells], raw)
    cutoff = decision_cutoff("2024-01-02")
    graph = freeze_graph(ledger, raw, cutoff, limit=8)
    examples = [(graph, cutoff + timedelta(days=i), "UP") for i in range(5)]
    rules, _ = mine_trace_rules(examples)
    relation_rules = [rule for rule in rules if rule["body"] == ["PARTNERED", "SELLS", "EXTRACTED_FROM"]]
    assert relation_rules
    context, _ = explore_trace(graph, cutoff, relation_rules)
    validate_context(graph, cutoff, context)
    assert len(context["entities"]) == 2
    assert {row["text"] for row in context["claims"]} == {"Alpha PARTNERED Beta", "Beta SELLS Widget"}
    text = context["evidences"][0]["text"]
    assert "Alpha partnered with Beta." in text and "Beta sells Widget." in text
    damaged = deepcopy(graph)
    edge = next(row for row in damaged["relations"] if row["type"] == "PARTNERED")
    edge["quotes"] = {}
    assert not explore_trace(damaged, cutoff, relation_rules)[0]["paths"]


def test_investment_reverse_traversal_keeps_forward_edge_direction():
    raw = source()
    p = proposal(raw, "INVESTED_IN", head={"kind": "Company", "name": "Beta"},
                 tail={"kind": "Company", "name": "Alpha", "ticker": "0700.HK"},
                 quote="Alpha invested in Beta.")
    # A semantic verifier should reject this deliberately reversed claim.
    ledger = GraphLedger("0700.HK", strict_cutoff=True)
    assert commit(ledger, [p], raw, supported=False)["rejected"]
    p = proposal(raw, "INVESTED_IN", quote="Alpha invested in Beta.")
    commit(ledger, [p], raw)
    graph = freeze_graph(ledger, raw, decision_cutoff("2024-01-02"), limit=8)
    store_edge = next(edge for edge in graph["relations"] if edge["type"] == "INVESTED_IN")
    assert any(kind == "^INVESTED_IN" and edge == store_edge
               for kind, node, edge in EvidenceGraph(graph, decision_cutoff("2024-01-02")).edges[store_edge["end_id"]])


def test_extraction_cache_reuses_only_verified_fixed_corpus(tmp_path):
    raw = source()
    p = proposal(raw)
    class Client:
        def __init__(self):
            self.calls = 0
            self.requests = []
        def generate_json(self, system, user):
            self.calls += 1
            self.requests.append((system, user))
            if json.loads(user)["task"] == "verify":
                return {"reviews": [{"proposal_index": 0, "supported": True, "evidence_ids": p.evidence_ids, "reason": "fixture"}]}
            return {"proposals": [json_value(p.model_dump())]}
    from src.eval.records import json_value
    client = Client()
    updater = BusinessUpdater(client, client)
    a = GraphLedger("0700.HK", strict_cutoff=True)
    updater.update(a, raw, decision_cutoff("2024-01-02"), cache_dir=tmp_path)
    b = GraphLedger("0700.HK", strict_cutoff=True)
    result = updater.update(b, raw, decision_cutoff("2024-01-03"), cache_dir=tmp_path)
    assert result["cache_hit"] and client.calls == 2
    assert a.state == b.state
    assert client.requests[0][0] == client.requests[1][0]
    prefix = client.requests[0][1].split(',"task":')[0]
    assert client.requests[1][1].startswith(prefix + ',"task":')
    assert "proposals" not in json.loads(client.requests[0][1])
    assert list(json.loads(client.requests[1][1]))[-2:] == ["task", "proposals"]
