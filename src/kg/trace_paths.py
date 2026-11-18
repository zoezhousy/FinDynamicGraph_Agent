"""TRACE-inspired bounded rule mining and rule-constrained evidence exploration.

Uses the project's company-rooted graph and heuristic relation selection. Mining
depths are 2–4; exploration uses beam 8, depth 3 and five terminal source paths.
The minimum support of five and mining sample cap are local recorded choices.
"""
from collections import Counter, defaultdict
from random import Random

from src.eval.protocol import available_time, AVAILABILITY
from src.eval.records import digest
from src.kg.thesis_paths import EvidenceGraph

VERSION = "trace_paths_v1"
POLICY = {"min_hops": 2, "mining_depth": 4, "exploration_depth": 3, "beam": 8,
          "top_paths": 5, "min_support": 5, "min_confidence": 0.60,
          "sample_width": 32, "seed": 0, "root": "company", "selector": "heuristic"}


def mine_trace_rules(examples):
    counts = defaultdict(Counter)
    seen_examples = set()
    sampled = 0
    for snapshot, cutoff, label in examples:
        graph = EvidenceGraph(snapshot, cutoff)
        key = (graph.root, cutoff.isoformat())
        if key in seen_examples or label not in {"UP", "DOWN"}:
            raise ValueError("Training requires unique stock/date examples and binary labels")
        seen_examples.add(key)
        frontier = [((graph.root,), (), ())]
        patterns = set()
        rng = Random(digest([POLICY["seed"], key]))
        for depth in range(1, POLICY["mining_depth"] + 1):
            candidates = [child for path in frontier if path[0][-1] not in graph.evidence
                          for child in graph.extend(path)]
            candidates.sort(key=digest)
            if len(candidates) > POLICY["sample_width"]:
                sampled += len(candidates) - POLICY["sample_width"]
                candidates = rng.sample(candidates, POLICY["sample_width"])
            frontier = candidates
            if depth >= POLICY["min_hops"]:
                patterns.update(body for nodes, body, _ in candidates if nodes[-1] in graph.evidence)
        for body in patterns:
            counts[body][label] += 1
    candidates = []
    for body, labels in sorted(counts.items()):
        support = sum(labels.values())
        for head in ("UP", "DOWN"):
            confidence = labels[head] / support
            if support >= POLICY["min_support"] and confidence >= POLICY["min_confidence"]:
                candidates.append({"body": list(body), "body_atoms": [
                    {"relation": relation, "start_var": f"X{i}", "end_var": f"X{i + 1}"}
                    for i, relation in enumerate(body)], "head": head, "support": support,
                    "confidence": confidence, "label_counts": dict(labels)})
    # In this linear path language a shorter rule subsumes an extension only
    # when it has the same head, at least its confidence, and identical support.
    rules = [rule for rule in candidates if not any(
        len(other["body"]) < len(rule["body"]) and rule["body"][:len(other["body"])] == other["body"]
        and other["head"] == rule["head"] and other["confidence"] >= rule["confidence"]
        and other["support"] == rule["support"] for other in candidates)]
    return rules, {"version": VERSION, "policy": POLICY, "n_examples": len(seen_examples),
                   "n_patterns": len(counts), "n_rules": len(rules), "sampled_out": sampled,
                   "rules_digest": digest(rules)}


def compact_context(snapshot, selected):
    """One copy of path facts and exact terminal quotes, without full graph rows."""
    nodes = {node for path in selected for node in path["node_ids"]}
    quotes = defaultdict(set)
    for path in selected:
        for edge in path["edges"]:
            if edge.get("claim_id"):
                nodes.add(edge["claim_id"])
                for eid, quote in edge.get("quotes", {}).items():
                    nodes.add(eid)
                    quotes[eid].add(quote)
            if edge["type"] in {"SUPPORTED_BY", "EXTRACTED_FROM"} and edge.get("quote"):
                quotes[edge["end_id"]].add(edge["quote"])
    evidences = [{"evidence_id": row["evidence_id"], "text": "\n".join(sorted(quotes[row["evidence_id"]])),
                  "published_at": row["published_at"], "source_type": row["source_type"]}
                 for row in sorted(snapshot["evidences"], key=lambda row: row["evidence_id"])
                 if row["evidence_id"] in nodes and quotes[row["evidence_id"]]]
    context = {"retrieval_version": VERSION, "paths": selected, "evidences": evidences,
            "claims": [{key: row[key] for key in ("entity_id", "text", "metric", "period", "polarity")}
                       for row in sorted(snapshot["claims"], key=lambda row: row["entity_id"]) if row["entity_id"] in nodes],
            "observations": [{key: row.get(key) for key in ("entity_id", "type", "period", "value", "unit")}
                             for row in sorted(snapshot.get("observations", []), key=lambda row: row["entity_id"])
                             if row["entity_id"] in nodes]}
    if snapshot.get("business_entities"):
        import json
        for claim in context["claims"]:
            if claim["metric"] == "business_relation_v1":
                relation = json.loads(claim["text"])
                claim["text"] = relation["head"]["name"] + " " + relation["relation"] + " " + relation["tail"]["name"]
        context["entities"] = [{key: row[key] for key in ("entity_id", "type", "name")}
                               for row in sorted(snapshot["business_entities"], key=lambda row: row["entity_id"])
                               if row["entity_id"] in nodes]
    return context


def explore_trace(snapshot, cutoff, rules):
    graph = EvidenceGraph(snapshot, cutoff)
    if any(rule["head"] not in {"UP", "DOWN"} or not 0 <= rule["confidence"] <= 1
           or not POLICY["min_hops"] <= len(rule["body"]) <= POLICY["mining_depth"] for rule in rules):
        raise ValueError("Invalid frozen rule library")
    admitted = [rule for rule in rules if rule["support"] >= POLICY["min_support"]
                and rule["confidence"] > POLICY["min_confidence"]]
    frontier = [((graph.root,), (), ())]
    selected, pruned = [], 0
    degrees = Counter(end for edges in graph.edges.values() for _, end, _ in edges)

    def matching(body):
        return [rule for rule in admitted if tuple(rule["body"][:len(body)]) == body]

    def rank(path):
        nodes, body, edges = path
        matches = matching(body)
        complete = any(len(rule["body"]) == len(body) for rule in matches) and nodes[-1] in graph.evidence
        coverage = max(len(body) / len(rule["body"]) for rule in matches)
        fresh = max(available_time(edge[key]).timestamp() for edge in edges for key in AVAILABILITY if edge.get(key))
        hub = sum(degrees[node] + len(graph.edges[node]) for node in nodes)
        return (-int(complete), -coverage, -fresh, hub, len(body), digest(path))

    for _ in range(POLICY["exploration_depth"]):
        candidates = [child for path in frontier for child in graph.extend(path) if matching(child[1])]
        candidates.sort(key=rank)
        pruned += max(0, len(candidates) - POLICY["beam"])
        frontier = []
        for nodes, body, edges in candidates[:POLICY["beam"]]:
            complete = [rule for rule in matching(body) if len(rule["body"]) == len(body)]
            if nodes[-1] in graph.evidence:
                for rule in complete:
                    selected.append({"path_id": digest([nodes, body, rule]), "node_ids": list(nodes),
                                     "relations": list(body), "edges": list(edges),
                                     "rule": {key: rule[key] for key in ("body", "head", "confidence", "support")}})
            else:
                frontier.append((nodes, body, edges))
    selected.sort(key=lambda path: (-path["rule"]["confidence"], -path["rule"]["support"], path["path_id"]))
    context = compact_context(snapshot, selected[:POLICY["top_paths"]])
    return context, {"version": VERSION, "matched_count": len(selected), "selected_count": len(context["paths"]),
                     "beam_pruned": pruned, "rules_digest": digest(rules),
                     "status": "success" if context["paths"] else "no_eligible_rule_paths"}


def validate_context(snapshot, cutoff, context):
    """Validate a saved selection against actual eligible edges and source quotes."""
    graph = EvidenceGraph(snapshot, cutoff)
    if len(context["paths"]) > POLICY["top_paths"]:
        raise ValueError("Selected context exceeds path budget")
    for path in context["paths"]:
        if not path["node_ids"] or path["node_ids"][0] != graph.root:
            raise ValueError("Path must start at the frozen company")
        frontier = [((graph.root,), (), ())]
        for kind, end, expected in zip(path["relations"], path["node_ids"][1:], path["edges"], strict=True):
            frontier = [child for state in frontier for child in graph.extend(state)
                        if child[1][-1] == kind and child[0][-1] == end and child[2][-1] == expected]
        rule = path["rule"]
        if (not frontier or path["node_ids"][-1] not in graph.evidence
                or not POLICY["min_hops"] <= len(path["relations"]) <= POLICY["exploration_depth"]
                or path["relations"] != rule["body"] or rule["confidence"] <= POLICY["min_confidence"]
                or rule["support"] < POLICY["min_support"]):
            raise ValueError("Invalid rule-grounded evidence path")
    if context != compact_context(snapshot, context["paths"]):
        raise ValueError("Context contains information outside selected paths")
