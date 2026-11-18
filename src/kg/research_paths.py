"""Rule-constrained traversal of a frozen research graph."""
from collections import defaultdict
import re

from src.eval.protocol import eligible_record


RULESET = "research_claim_paths_v1"
_RULE_ORDER = {"conflicting_support": 0, "direct_support": 1, "shared_source_support": 2}
ABLATION_RULES = ("conflicting_support", "shared_source_support")


def _terms(text):
    words = re.findall(r"[\u3400-\u9fff]+|[^\W_\u3400-\u9fff]+", text.casefold())
    terms = set(words)
    for word in words:
        if "\u3400" <= word[0] <= "\u9fff":
            terms.update(word[i:i + 2] for i in range(len(word) - 1))
    return terms


def retrieve_research_evidence(snapshot, ticker, cutoff, query, *, top_k=4, seed_limit=4, disabled_rule=None):
    """Compare lexical text, direct support, and two bounded multi-hop rules."""
    terms = _terms(query)
    if not terms or top_k < 1 or seed_limit < 1:
        raise ValueError("A lexical query and positive retrieval limits are required")
    if disabled_rule is not None and disabled_rule not in ABLATION_RULES:
        raise ValueError("Only multi-hop rules can be disabled for ablation")

    def index(rows, field):
        indexed = {row[field]: row for row in rows}
        if len(indexed) != len(rows) or any(
            row.get("ticker") != ticker or not eligible_record(row, cutoff) for row in rows
        ):
            raise ValueError("Snapshot contains duplicate, unavailable or out-of-scope nodes")
        return indexed

    evidence = index(snapshot["evidences"], "evidence_id")
    claims = index(snapshot["claims"], "entity_id")
    if set(evidence) & set(claims):
        raise ValueError("Evidence and claim identities must be distinct")

    support, supported_claims, conflicts = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    for edge in snapshot["relations"]:
        start, end, kind = edge["start_id"], edge["end_id"], edge["type"]
        if (not edge.get("effective_at") or not eligible_record(edge, cutoff)
                or start not in claims or end not in (claims.keys() | evidence.keys())):
            raise ValueError("Snapshot contains an unavailable or dangling relation")
        if kind == "SUPPORTED_BY":
            item = evidence.get(end)
            quote = claims[start].get("quotes", {}).get(end, "").strip()
            if (item is None or end not in claims[start]["evidence_ids"]
                    or claims[start].get("verification", {}).get("supported") is not True
                    or not quote or quote not in item["text"]):
                raise ValueError("Support relation requires a verified claim and an exact quote")
            witness = {**edge, "quote": quote}
            support[start][end] = witness
            supported_claims[end][start] = witness
        elif kind == "CONFLICTS_WITH":
            if end not in claims or start == end or claims[start]["comparison_key"] != claims[end]["comparison_key"]:
                raise ValueError("Conflict relation must link distinct claims about the same fact")
            conflicts[start][end] = conflicts[end][start] = edge

    def ranked_ids(records, fields):
        scores = {
            ident: len(terms & _terms(" ".join(str(row.get(field) or "") for field in fields)))
            for ident, row in records.items()
        }
        return sorted((i for i in records if scores[i]), key=lambda i: (-scores[i], i)), scores

    text_ids, text_scores = ranked_ids(evidence, ("text",))
    seeds, seed_scores = ranked_ids(claims, ("subject", "metric", "period", "text"))
    seeds = seeds[:seed_limit]
    direct, multihop = {}, {}

    def add_path(rule, nodes, edges):
        if rule == disabled_rule:
            return
        seed, target = nodes[0], nodes[-1]
        rank = (-seed_scores[seed], _RULE_ORDER[rule], tuple(nodes))
        path = {"rule": rule, "nodes": nodes, "edges": edges, "hops": len(edges),
                "seed_score": seed_scores[seed], "evidence_id": target,
                "claims": [{field: claims[node][field] for field in ("entity_id", "metric", "period", "text")}
                           for node in nodes if node in claims]}
        for pool in (direct, multihop) if rule == "direct_support" else (multihop,):
            if target not in pool or rank < pool[target][0]:
                pool[target] = (rank, path)

    for seed in seeds:
        for related, conflict in conflicts[seed].items():
            for target, edge in support[related].items():
                add_path("conflicting_support", [seed, related, target], [conflict, edge])
        for anchor, edge in support[seed].items():
            add_path("direct_support", [seed, anchor], [edge])
            for related, reverse_edge in supported_claims[anchor].items():
                if related != seed:
                    for target, target_edge in support[related].items():
                        if target != anchor:
                            add_path("shared_source_support", [seed, anchor, related, target],
                                     [edge, reverse_edge, target_edge])

    results = {"text_lexical": {
        "evidence_ids": text_ids[:top_k],
        "lexical_scores": {i: text_scores[i] for i in text_ids[:top_k]},
        "candidate_count": len(text_ids), "paths": [],
    }}
    for name, pool in (("graph_one_hop", direct), ("graph_multihop", multihop)):
        ranked = sorted(pool.values(), key=lambda entry: entry[0])[:top_k]
        paths = [path for _, path in ranked]
        results[name] = {
            "evidence_ids": [path["evidence_id"] for path in paths],
            "candidate_count": len(pool),
            "paths": paths,
            "seed_claim_ids": seeds,
            "new_beyond_direct_support": [path["evidence_id"] for path in paths if path["evidence_id"] not in direct],
        }
    return results
