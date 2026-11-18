"""Typed point-in-time paths and frozen rules for the dissertation experiment."""
from collections import Counter, defaultdict
from itertools import islice
from random import Random

from src.eval.protocol import eligible_record
from src.eval.records import digest
from src.kg.business_relations import RELATIONS as BUSINESS_RELATIONS

MIN_SUPPORT = 5
MIN_CONFIDENCE = 0.60
MAX_HOPS = 4
MINING_SAMPLE_WIDTH = 32
MINING_SEED = 0
BEAM_WIDTH = 8
PATH_LIMIT = 4


def freeze_graph(ledger, sources, cutoff, *, limit):
    metadata = {row["evidence_id"]: row for row in sources["evidences"]}
    snapshot = ledger.freeze(cutoff, limit=limit, active_evidence_ids=metadata)
    root = "company:" + ledger.ticker
    snapshot["company"][0]["entity_id"] = root
    snapshot["observations"] = []
    snapshot["securities"] = []
    securities = set()
    for row in snapshot["evidences"]:
        details = metadata[row["evidence_id"]]
        row.update({key: details[key] for key in ("source", "document_id", "version", "language", "source_fingerprint")})
        kind = row["source_type"]
        if kind not in {"southbound_holding", "ah_premium"}:
            continue
        observation = "observation:" + row["source_fingerprint"]
        snapshot["observations"].append({"entity_id": observation, "type": kind,
                                         "effective_at": row["published_at"],
                                         **{key: details[key] for key in ("period", "value", "unit", "a_share")}})
        if kind == "ah_premium":
            share = "security:" + details["a_share"]
            if share not in securities:
                snapshot["securities"].append({"entity_id": share, "ticker": details["a_share"]})
                securities.add(share)
            snapshot["relations"].append({"type": "AH_PREMIUM", "start_id": root, "end_id": share,
                                          "observation_id": observation, "effective_at": row["published_at"],
                                          "evidence_id": row["evidence_id"]})
            start, relation = share, "HAS_PREMIUM_OBSERVATION"
        else:
            start, relation = root, "SOUTHBOUND_HOLDING"
        snapshot["relations"].extend([
            {"type": relation, "start_id": start, "end_id": observation, "effective_at": row["published_at"],
             "evidence_id": row["evidence_id"]},
            {"type": "EXTRACTED_FROM", "start_id": observation, "end_id": row["evidence_id"],
             "effective_at": row["published_at"], "quote": row["text"]},
        ])
    for claim in snapshot["claims"]:
        snapshot["relations"].append({"type": "HAS_CLAIM", "start_id": root, "end_id": claim["entity_id"],
                                      "effective_at": claim["effective_at"]})
        if claim.get("metric") == "business_relation_v1" and claim.get("predicate") in BUSINESS_RELATIONS:
            # Source-owner provenance, not a parent/subsidiary identity assertion.
            # Preserve the actual transaction endpoints on the business edge.
            snapshot["relations"].append({"type": "HAS_VERIFIED_" + claim["predicate"] + "_CLAIM",
                "start_id": root, "end_id": claim["entity_id"], "effective_at": claim["effective_at"]})
    if any(row.get("metric") == "business_relation_v1" for row in snapshot["claims"]):
        from src.kg.business_relations import project_business
        business = project_business(snapshot, root)
        snapshot["business_entities"] = business["entities"]
        snapshot["relations"].extend(business["edges"])
    return snapshot


class EvidenceGraph:
    def __init__(self, snapshot, cutoff):
        company, = snapshot["company"]
        self.root = company["entity_id"]
        self.evidence = {row["evidence_id"]: row for row in snapshot["evidences"]
                         if eligible_record(row, cutoff, strict=True)}
        self.nodes = {self.root: company, **self.evidence}
        for key in ("claims", "observations"):
            for row in snapshot[key]:
                if eligible_record(row, cutoff, strict=True):
                    self.nodes[row["entity_id"]] = row
        self.nodes.update((row["entity_id"], row) for row in snapshot["securities"])
        self.nodes.update((row["entity_id"], row) for row in snapshot.get("business_entities", [])
                          if eligible_record(row, cutoff, strict=True))
        self.edges = defaultdict(list)
        for edge in snapshot["relations"]:
            start, end = edge["start_id"], edge["end_id"]
            if edge["type"].startswith("HAS_VERIFIED_"):
                claim = self.nodes.get(end, {})
                if (start != self.root or claim.get("metric") != "business_relation_v1"
                        or claim.get("predicate") not in BUSINESS_RELATIONS
                        or edge["type"] != "HAS_VERIFIED_" + claim["predicate"] + "_CLAIM"):
                    continue
            if start not in self.nodes or end not in self.nodes or not eligible_record(edge, cutoff, strict=True):
                continue
            if edge.get("evidence_id", end) not in self.nodes or (
                "observation_id" in edge and edge["observation_id"] not in self.nodes
            ):
                continue
            kind = edge["type"]
            if kind in BUSINESS_RELATIONS or (kind == "EXTRACTED_FROM" and edge.get("claim_id")):
                claim = self.nodes.get(edge.get("claim_id"), {})
                if (claim.get("metric") != "business_relation_v1" or not edge.get("evidence_ids")
                        or set(edge["evidence_ids"]) != set(claim.get("evidence_ids", []))
                        or edge.get("quotes") != claim.get("quotes")
                        or any(eid not in self.evidence or not quote or quote not in self.evidence[eid]["text"]
                               for eid, quote in edge.get("quotes", {}).items())):
                    continue
                endpoints = (claim.get("subject"), claim.get("value"))
                if kind in BUSINESS_RELATIONS:
                    if kind != claim.get("predicate") or (start, end) != endpoints:
                        continue
                elif start not in endpoints or end not in claim["evidence_ids"] or edge.get("quote") != claim["quotes"].get(end):
                    continue
            if kind in {"SUPPORTED_BY", "EXTRACTED_FROM"}:
                quote = edge.get("quote") if kind == "EXTRACTED_FROM" else self.nodes[start]["quotes"].get(end)
                if end not in self.evidence or not quote or quote not in self.evidence[end]["text"]:
                    continue
                edge = {**edge, "quote": quote}
            self.edges[start].append((kind, end, edge))
            if kind in {"CONFLICTS_WITH", "SUPPORTED_BY"}:
                self.edges[end].append(("^" + kind, start, edge))
            if kind in BUSINESS_RELATIONS:
                self.edges[end].append((kind if kind == "PARTNERED" else "^" + kind, start, edge))
        for edges in self.edges.values():
            edges.sort(key=lambda item: (item[0], item[1], digest(item[2])))

    def extend(self, path):
        nodes, body, edges = path
        for kind, node, edge in self.edges[nodes[-1]]:
            if kind == "HAS_PREMIUM_OBSERVATION" and edges[-1].get("observation_id") != node:
                continue
            if kind == "EXTRACTED_FROM" and edge.get("claim_id"):
                if not edges or edges[-1].get("claim_id") != edge["claim_id"]:
                    continue
            if node not in nodes:
                yield (nodes + (node,), body + (kind,), edges + (edge,))


def mine_rules(examples):
    """Sample training paths and count each relation pattern once per example."""
    counts = defaultdict(Counter)
    for snapshot, cutoff, label in examples:
        if label not in {"UP", "DOWN"}:
            raise ValueError("Rule example requires an UP or DOWN label")
        graph = EvidenceGraph(snapshot, cutoff)
        frontier = [((graph.root,), (), ())]
        rng = Random(digest([MINING_SEED, graph.root, cutoff.isoformat()]))
        seen = set()
        for depth in range(1, MAX_HOPS + 1):
            candidates = [child for path in frontier for child in graph.extend(path)]
            if not candidates:
                break
            if len(candidates) > MINING_SAMPLE_WIDTH:
                candidates = rng.sample(candidates, MINING_SAMPLE_WIDTH)
            frontier = candidates
            if depth >= 2:
                seen.update(body for nodes, body, _ in frontier if nodes[-1] in graph.evidence)
        for body in seen:
            counts[body][label] += 1
    rules = []
    for body, directions in sorted(counts.items()):
        head, count = directions.most_common(1)[0]
        support = sum(directions.values())
        confidence = count / support
        if support >= MIN_SUPPORT and confidence >= MIN_CONFIDENCE:
            # Adjacent atoms share a variable. The retrieval index uses their ordered relation types.
            atoms = [{"relation": relation, "start_var": f"X{i}", "end_var": f"X{i + 1}"}
                     for i, relation in enumerate(body)]
            rules.append({"body": list(body), "body_atoms": atoms, "head": head, "support": support,
                          "confidence": confidence, "label_counts": dict(directions)})
    return rules


def retrieve_paths(snapshot, cutoff, rules):
    graph = EvidenceGraph(snapshot, cutoff)
    by_body = {tuple(rule["body"]): rule for rule in rules}
    prefixes = {}
    for body, rule in by_body.items():
        priority = (-rule["confidence"], -rule["support"])
        for depth in range(1, len(body) + 1):
            prefix = body[:depth]
            prefixes[prefix] = min(priority, prefixes.get(prefix, priority))

    frontier = [((graph.root,), (), ())]
    matches, pruned = [], 0
    for _ in range(MAX_HOPS):
        candidates = [child for path in frontier for child in graph.extend(path) if child[1] in prefixes]
        candidates.sort(key=lambda path: (prefixes[path[1]], digest(path)))
        pruned += max(0, len(candidates) - BEAM_WIDTH)
        frontier = candidates[:BEAM_WIDTH]
        for path in frontier:
            nodes, body, edges = path
            if body in by_body and nodes[-1] in graph.evidence:
                matches.append({"path_id": digest(path), "node_ids": nodes,
                                "edges": edges, "rule": by_body[body]})
    matches.sort(key=lambda match: (-match["rule"]["confidence"], -match["rule"]["support"], match["path_id"]))
    selected = list(islice(matches, PATH_LIMIT))
    ids = {node for path in selected for node in path["node_ids"]}
    # Every intermediate evidence node is also available to the analysts.
    evidence = [item for eid, item in sorted(graph.evidence.items()) if eid in ids]
    claims = [{key: claim[key] for key in ("entity_id", "text", "metric", "period", "value", "polarity")}
              for claim in snapshot["claims"] if claim["entity_id"] in ids]
    context = {"paths": selected, "claims": claims, "evidences": evidence,
               "observations": [row for row in snapshot["observations"] if row["entity_id"] in ids]}
    return context, {"matched_count": len(matches), "beam_pruned": pruned,
                     "output_truncated": max(0, len(matches) - PATH_LIMIT)}
