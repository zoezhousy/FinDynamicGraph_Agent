from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from neo4j import GraphDatabase, basic_auth


class KGQueryClient:
    """Read-only query helper for agents."""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        self._driver = GraphDatabase.driver(
            uri,
            auth=basic_auth(user, password),
            notifications_disabled_categories=["UNRECOGNIZED"],
        )
        self._database = database

    def close(self) -> None:
        driver, self._driver = self._driver, None
        if driver is not None:
            driver.close()

    def load_subgraph(
        self,
        ticker: str,
        as_of_date: datetime,
        max_news: int = 20,
        max_signals: int = 50,
        signal_window_days: int = 90,
        news_window_days: int = 30,
    ) -> dict[str, list[dict[str, Any]]]:
        """Load the ticker subgraph exactly as it was valid at ``as_of_date``.

        Avoids future leakage by enforcing:
        - valid_from <= as_of_date
        - valid_to IS NULL OR valid_to > as_of_date
        - effective_at IS NULL OR effective_at <= as_of_date
        - published_at <= as_of_date (for evidence/news)

        ``is_active`` deliberately is not used here.  It describes the graph's
        current state and would incorrectly hide a record that was active at a
        historical decision date but expired later.
        """
        signal_from = as_of_date - timedelta(days=signal_window_days)
        news_from = as_of_date - timedelta(days=news_window_days)
        as_of_iso = _end_of_day(as_of_date).isoformat()
        signal_from_iso = signal_from.isoformat()
        news_from_iso = news_from.isoformat()

        cypher = """
        MATCH (c:Company {ticker: $ticker})
        // Aggregate independent branches before joining them.
        CALL {
          WITH c
          OPTIONAL MATCH (c)-[r1:HAS_SIGNAL]->(s:IndicatorSignal)
        WHERE datetime(r1.as_of_date) <= datetime($as_of)
          AND datetime(r1.as_of_date) >= datetime($signal_from)
          AND (r1.valid_from IS NULL OR datetime(r1.valid_from) <= datetime($as_of))
          AND (r1.valid_to IS NULL OR datetime(r1.valid_to) > datetime($as_of))
          AND (r1.effective_at IS NULL OR datetime(r1.effective_at) <= datetime($as_of))
          AND (s.valid_from IS NULL OR datetime(s.valid_from) <= datetime($as_of))
          AND (s.valid_to IS NULL OR datetime(s.valid_to) > datetime($as_of))
          AND (s.effective_at IS NULL OR datetime(s.effective_at) <= datetime($as_of))

        // ── Fundamentals ──
        
          RETURN collect(DISTINCT s) AS raw_signals
        }
        CALL {
          WITH c
          OPTIONAL MATCH (c)-[rf:HAS_SIGNAL]->(f:FundamentalSignal)
        WHERE datetime(rf.as_of_date) <= datetime($as_of)
          AND (rf.valid_from IS NULL OR datetime(rf.valid_from) <= datetime($as_of))
          AND (rf.valid_to IS NULL OR datetime(rf.valid_to) > datetime($as_of))
          AND (rf.effective_at IS NULL OR datetime(rf.effective_at) <= datetime($as_of))
          AND (f.valid_from IS NULL OR datetime(f.valid_from) <= datetime($as_of))
          AND (f.valid_to IS NULL OR datetime(f.valid_to) > datetime($as_of))
          AND (f.effective_at IS NULL OR datetime(f.effective_at) <= datetime($as_of))

        // ── Risks ──
        OPTIONAL MATCH (f)-[rfe:SUPPORTED_BY]->(fe:Evidence)
          WHERE (fe.published_at IS NULL OR datetime(fe.published_at) <= datetime($as_of))
            AND (fe.valid_from IS NULL OR datetime(fe.valid_from) <= datetime($as_of))
            AND (fe.valid_to IS NULL OR datetime(fe.valid_to) > datetime($as_of))
            AND (fe.effective_at IS NULL OR datetime(fe.effective_at) <= datetime($as_of))
            AND (rfe.valid_from IS NULL OR datetime(rfe.valid_from) <= datetime($as_of))
            AND (rfe.valid_to IS NULL OR datetime(rfe.valid_to) > datetime($as_of))
            AND (rfe.effective_at IS NULL OR datetime(rfe.effective_at) <= datetime($as_of))
        OPTIONAL MATCH (src_fund:SourceDocument)-[rsf:CONTAINS_EVIDENCE]->(fe)
          WHERE (src_fund.valid_from IS NULL OR datetime(src_fund.valid_from) <= datetime($as_of))
            AND (src_fund.valid_to IS NULL OR datetime(src_fund.valid_to) > datetime($as_of))
            AND (src_fund.effective_at IS NULL OR datetime(src_fund.effective_at) <= datetime($as_of))
            AND (rsf.valid_from IS NULL OR datetime(rsf.valid_from) <= datetime($as_of))
            AND (rsf.valid_to IS NULL OR datetime(rsf.valid_to) > datetime($as_of))
            AND (rsf.effective_at IS NULL OR datetime(rsf.effective_at) <= datetime($as_of))
        OPTIONAL MATCH (fe)-[rfc:SUPPORTS_CLAIM|CONTRADICTS_CLAIM]->(fc:Claim)
          WHERE (fc.valid_from IS NULL OR datetime(fc.valid_from) <= datetime($as_of))
            AND (fc.valid_to IS NULL OR datetime(fc.valid_to) > datetime($as_of))
            AND (fc.effective_at IS NULL OR datetime(fc.effective_at) <= datetime($as_of))
            AND (rfc.valid_from IS NULL OR datetime(rfc.valid_from) <= datetime($as_of))
            AND (rfc.valid_to IS NULL OR datetime(rfc.valid_to) > datetime($as_of))
            AND (rfc.effective_at IS NULL OR datetime(rfc.effective_at) <= datetime($as_of))
        OPTIONAL MATCH (fc)-[cf2:CONFLICTS_WITH]->(other2:Claim)
          WHERE other2.entity_id <> fc.entity_id
            AND (other2.valid_from IS NULL OR datetime(other2.valid_from) <= datetime($as_of))
            AND (other2.valid_to IS NULL OR datetime(other2.valid_to) > datetime($as_of))
            AND (other2.effective_at IS NULL OR datetime(other2.effective_at) <= datetime($as_of))
            AND (cf2.valid_from IS NULL OR datetime(cf2.valid_from) <= datetime($as_of))
            AND (cf2.valid_to IS NULL OR datetime(cf2.valid_to) > datetime($as_of))
            AND (cf2.effective_at IS NULL OR datetime(cf2.effective_at) <= datetime($as_of))
        
          RETURN collect(DISTINCT f) AS fund_nodes,
                 collect(DISTINCT fe) AS fund_evidences,
                 collect(DISTINCT src_fund) AS fund_sources,
                 collect(DISTINCT fc) AS fund_claims,
                 collect(DISTINCT {a: fc.entity_id, b: other2.entity_id}) AS fund_conflicts
        }
        CALL {
          WITH c
          OPTIONAL MATCH (c)-[rr:HAS_RISK]->(r:RiskEvent)
        WHERE datetime(rr.as_of_date) <= datetime($as_of)
          AND (rr.valid_from IS NULL OR datetime(rr.valid_from) <= datetime($as_of))
          AND (rr.valid_to IS NULL OR datetime(rr.valid_to) > datetime($as_of))
          AND (rr.effective_at IS NULL OR datetime(rr.effective_at) <= datetime($as_of))
          AND (r.valid_from IS NULL OR datetime(r.valid_from) <= datetime($as_of))
          AND (r.valid_to IS NULL OR datetime(r.valid_to) > datetime($as_of))
          AND (r.effective_at IS NULL OR datetime(r.effective_at) <= datetime($as_of))

        // ── News ──
        OPTIONAL MATCH (r)-[rre:SUPPORTED_BY]->(re:Evidence)
          WHERE (re.published_at IS NULL OR datetime(re.published_at) <= datetime($as_of))
            AND (re.valid_from IS NULL OR datetime(re.valid_from) <= datetime($as_of))
            AND (re.valid_to IS NULL OR datetime(re.valid_to) > datetime($as_of))
            AND (re.effective_at IS NULL OR datetime(re.effective_at) <= datetime($as_of))
            AND (rre.valid_from IS NULL OR datetime(rre.valid_from) <= datetime($as_of))
            AND (rre.valid_to IS NULL OR datetime(rre.valid_to) > datetime($as_of))
            AND (rre.effective_at IS NULL OR datetime(rre.effective_at) <= datetime($as_of))

        // ── Sources ──
        OPTIONAL MATCH (src_risk:SourceDocument)-[rsr:CONTAINS_EVIDENCE]->(re)
          WHERE (src_risk.valid_from IS NULL OR datetime(src_risk.valid_from) <= datetime($as_of))
            AND (src_risk.valid_to IS NULL OR datetime(src_risk.valid_to) > datetime($as_of))
            AND (src_risk.effective_at IS NULL OR datetime(src_risk.effective_at) <= datetime($as_of))
            AND (rsr.valid_from IS NULL OR datetime(rsr.valid_from) <= datetime($as_of))
            AND (rsr.valid_to IS NULL OR datetime(rsr.valid_to) > datetime($as_of))
            AND (rsr.effective_at IS NULL OR datetime(rsr.effective_at) <= datetime($as_of))

        // ── Claims ──
        OPTIONAL MATCH (re)-[rrc:SUPPORTS_CLAIM|CONTRADICTS_CLAIM]->(rc:Claim)
          WHERE (rc.valid_from IS NULL OR datetime(rc.valid_from) <= datetime($as_of))
            AND (rc.valid_to IS NULL OR datetime(rc.valid_to) > datetime($as_of))
            AND (rc.effective_at IS NULL OR datetime(rc.effective_at) <= datetime($as_of))
            AND (rrc.valid_from IS NULL OR datetime(rrc.valid_from) <= datetime($as_of))
            AND (rrc.valid_to IS NULL OR datetime(rrc.valid_to) > datetime($as_of))
            AND (rrc.effective_at IS NULL OR datetime(rrc.effective_at) <= datetime($as_of))

        // ── Conflicts between claims ──
        OPTIONAL MATCH (rc)-[cf3:CONFLICTS_WITH]->(other3:Claim)
          WHERE other3.entity_id <> rc.entity_id
            AND (other3.valid_from IS NULL OR datetime(other3.valid_from) <= datetime($as_of))
            AND (other3.valid_to IS NULL OR datetime(other3.valid_to) > datetime($as_of))
            AND (other3.effective_at IS NULL OR datetime(other3.effective_at) <= datetime($as_of))
            AND (cf3.valid_from IS NULL OR datetime(cf3.valid_from) <= datetime($as_of))
            AND (cf3.valid_to IS NULL OR datetime(cf3.valid_to) > datetime($as_of))
            AND (cf3.effective_at IS NULL OR datetime(cf3.effective_at) <= datetime($as_of))

          RETURN collect(DISTINCT r) AS risk_nodes,
                 collect(DISTINCT re) AS risk_evidences,
                 collect(DISTINCT src_risk) AS risk_sources,
                 collect(DISTINCT rc) AS risk_claims,
                 collect(DISTINCT {a: rc.entity_id, b: other3.entity_id}) AS risk_conflicts
        }
        CALL {
          WITH c
          OPTIONAL MATCH (c)-[r2:MENTIONED_IN]->(n:NewsEvent)
        WHERE datetime(r2.as_of_date) <= datetime($as_of)
          AND datetime(r2.as_of_date) >= datetime($news_from)
          AND (r2.valid_from IS NULL OR datetime(r2.valid_from) <= datetime($as_of))
          AND (r2.valid_to IS NULL OR datetime(r2.valid_to) > datetime($as_of))
          AND (r2.effective_at IS NULL OR datetime(r2.effective_at) <= datetime($as_of))
          AND (n.valid_from IS NULL OR datetime(n.valid_from) <= datetime($as_of))
          AND (n.valid_to IS NULL OR datetime(n.valid_to) > datetime($as_of))
          AND (n.effective_at IS NULL OR datetime(n.effective_at) <= datetime($as_of))

        // ── Evidence via SUPPORTED_BY ──
        OPTIONAL MATCH (n)-[rne:SUPPORTED_BY]->(ne:Evidence)
          WHERE (ne.published_at IS NULL OR datetime(ne.published_at) <= datetime($as_of))
            AND (ne.valid_from IS NULL OR datetime(ne.valid_from) <= datetime($as_of))
            AND (ne.valid_to IS NULL OR datetime(ne.valid_to) > datetime($as_of))
            AND (ne.effective_at IS NULL OR datetime(ne.effective_at) <= datetime($as_of))
            AND (rne.valid_from IS NULL OR datetime(rne.valid_from) <= datetime($as_of))
            AND (rne.valid_to IS NULL OR datetime(rne.valid_to) > datetime($as_of))
            AND (rne.effective_at IS NULL OR datetime(rne.effective_at) <= datetime($as_of))
        OPTIONAL MATCH (src_news:SourceDocument)-[rsn:CONTAINS_EVIDENCE]->(ne)
          WHERE (src_news.valid_from IS NULL OR datetime(src_news.valid_from) <= datetime($as_of))
            AND (src_news.valid_to IS NULL OR datetime(src_news.valid_to) > datetime($as_of))
            AND (src_news.effective_at IS NULL OR datetime(src_news.effective_at) <= datetime($as_of))
            AND (rsn.valid_from IS NULL OR datetime(rsn.valid_from) <= datetime($as_of))
            AND (rsn.valid_to IS NULL OR datetime(rsn.valid_to) > datetime($as_of))
            AND (rsn.effective_at IS NULL OR datetime(rsn.effective_at) <= datetime($as_of))
        OPTIONAL MATCH (ne)-[rnc:SUPPORTS_CLAIM|CONTRADICTS_CLAIM]->(nc:Claim)
          WHERE (nc.valid_from IS NULL OR datetime(nc.valid_from) <= datetime($as_of))
            AND (nc.valid_to IS NULL OR datetime(nc.valid_to) > datetime($as_of))
            AND (nc.effective_at IS NULL OR datetime(nc.effective_at) <= datetime($as_of))
            AND (rnc.valid_from IS NULL OR datetime(rnc.valid_from) <= datetime($as_of))
            AND (rnc.valid_to IS NULL OR datetime(rnc.valid_to) > datetime($as_of))
            AND (rnc.effective_at IS NULL OR datetime(rnc.effective_at) <= datetime($as_of))
        OPTIONAL MATCH (nc)-[cf1:CONFLICTS_WITH]->(other1:Claim)
          WHERE other1.entity_id <> nc.entity_id
            AND (other1.valid_from IS NULL OR datetime(other1.valid_from) <= datetime($as_of))
            AND (other1.valid_to IS NULL OR datetime(other1.valid_to) > datetime($as_of))
            AND (other1.effective_at IS NULL OR datetime(other1.effective_at) <= datetime($as_of))
            AND (cf1.valid_from IS NULL OR datetime(cf1.valid_from) <= datetime($as_of))
            AND (cf1.valid_to IS NULL OR datetime(cf1.valid_to) > datetime($as_of))
            AND (cf1.effective_at IS NULL OR datetime(cf1.effective_at) <= datetime($as_of))
        
          RETURN collect(DISTINCT n) AS news_nodes,
                 collect(DISTINCT ne) AS news_evidences,
                 collect(DISTINCT src_news) AS news_sources,
                 collect(DISTINCT nc) AS news_claims,
                 collect(DISTINCT {a: nc.entity_id, b: other1.entity_id}) AS news_conflicts
        }
        RETURN c,
               raw_signals[0..$max_signals] AS signals,
               fund_nodes AS fundamentals,
               risk_nodes AS risks,
               news_nodes[0..$max_news] AS news,
               news_evidences + fund_evidences + risk_evidences AS evidences,
               news_sources + fund_sources + risk_sources AS sources,
               news_claims + fund_claims + risk_claims AS claims,
               news_conflicts + fund_conflicts + risk_conflicts AS conflict_pairs
        """

        with self._driver.session(database=self._database) as session:
            rec = session.run(
                cypher,
                ticker=ticker,
                as_of=as_of_iso,
                signal_from=signal_from_iso,
                news_from=news_from_iso,
                max_news=max_news,
                max_signals=max_signals,
            ).single()

            if not rec:
                return _empty_subgraph()

            company = rec["c"]
            signals = rec["signals"] or []
            fundamentals = rec["fundamentals"] or []
            risks = rec["risks"] or []
            news = rec["news"] or []
            evidences = rec["evidences"] or []
            sources = rec["sources"] or []
            claims = rec["claims"] or []
            conflict_pairs = rec["conflict_pairs"] or []

            signal_rows = [_clean_node(dict(node)) for node in signals if node]
            fundamental_rows = [_clean_node(dict(node)) for node in fundamentals if node]
            risk_rows = [_clean_node(dict(node)) for node in risks if node]
            news_rows = [_clean_node(dict(node)) for node in news if node]
            evidence_rows = [_clean_node(dict(node)) for node in evidences if node]
            source_rows = [_clean_node(dict(node)) for node in sources if node]
            claim_rows = [_clean_node(dict(node)) for node in claims if node]

            # Deduplicate conflicts
            conflict_set: set[tuple[str, str]] = set()
            for pair in conflict_pairs:
                a_id = pair.get("a")
                b_id = pair.get("b")
                if a_id and b_id and a_id != b_id:
                    key = tuple(sorted([a_id, b_id]))
                    conflict_set.add(key)  # type: ignore[arg-type]
            conflicts = [{"claim_a": a, "claim_b": b} for a, b in sorted(conflict_set)]

            def date_sort_key(x: dict[str, Any]) -> str:
                return str(
                    x.get("as_of_date")
                    or x.get("published_at")
                    or x.get("valid_from")
                    or ""
                )

            signal_rows = sorted(signal_rows, key=date_sort_key, reverse=True)[:max_signals]
            fundamental_rows = sorted(fundamental_rows, key=date_sort_key, reverse=True)
            risk_rows = sorted(risk_rows, key=date_sort_key, reverse=True)
            news_rows = sorted(news_rows, key=date_sort_key, reverse=True)[:max_news]
            evidence_rows = sorted(evidence_rows, key=date_sort_key, reverse=True)
            source_rows = sorted(source_rows, key=date_sort_key, reverse=True)
            claim_rows = sorted(claim_rows, key=date_sort_key, reverse=True)

            return {
                "company": [_clean_node(dict(company))] if company else [],
                "signals": signal_rows,
                "fundamentals": fundamental_rows,
                "risks": risk_rows,
                "news": news_rows,
                "evidences": evidence_rows,
                "sources": source_rows,
                "claims": claim_rows,
                "conflicts": conflicts,
            }

    def get_ticker_subgraph(
        self,
        ticker: str,
        as_of_date: datetime,
        max_news: int = 20,
        max_signals: int = 50,
        signal_window_days: int = 90,
        news_window_days: int = 30,
    ) -> dict[str, list[dict[str, Any]]]:
        """Backward-compatible alias for :meth:`load_subgraph`."""
        return self.load_subgraph(
            ticker=ticker,
            as_of_date=as_of_date,
            max_news=max_news,
            max_signals=max_signals,
            signal_window_days=signal_window_days,
            news_window_days=news_window_days,
        )

    def get_snapshot_summary(
        self,
        ticker: str,
        as_of_date: datetime,
    ) -> dict[str, Any]:
        """Summarise the same canonical point-in-time subgraph agents use."""
        subgraph = self.load_subgraph(ticker, as_of_date)
        return compute_snapshot_summary_from_lists(
            ticker=ticker,
            as_of_date=as_of_date,
            claims=subgraph.get("claims", []),
            evidences=subgraph.get("evidences", []),
            signals=(
                subgraph.get("signals", [])
                + subgraph.get("fundamentals", [])
                + subgraph.get("risks", [])
            ),
        )

    def _get_snapshot_summary_legacy(
        self,
        ticker: str,
        as_of_date: datetime,
    ) -> dict[str, Any]:
        """Return a summary of the KG snapshot at a given date.

        Avoids future leakage with the same temporal filters as get_ticker_subgraph.
        """
        as_of_iso = _end_of_day(as_of_date).isoformat()

        cypher = """
        MATCH (c:Company {ticker: $ticker})

        // ── Active claims ──
        OPTIONAL MATCH (c)-[:HAS_SIGNAL|HAS_RISK|MENTIONED_IN*1..3]->(cl:Claim {ticker: $ticker})
        WHERE (cl.valid_from IS NULL OR datetime(cl.valid_from) <= datetime($as_of))
          AND (cl.valid_to IS NULL OR datetime(cl.valid_to) > datetime($as_of))
          AND (cl.is_active IS NULL OR cl.is_active = true)

        // ── Active signals ──
        OPTIONAL MATCH (c)-[r_sig:HAS_SIGNAL]->(sig)
        WHERE (sig:IndicatorSignal OR sig:FundamentalSignal)
          AND (r_sig.valid_from IS NULL OR datetime(r_sig.valid_from) <= datetime($as_of))
          AND (r_sig.valid_to IS NULL OR datetime(r_sig.valid_to) > datetime($as_of))
          AND (r_sig.is_active IS NULL OR r_sig.is_active = true)

        // ── Evidence ──
        OPTIONAL MATCH (ev:Evidence)
        WHERE (ev.evidence_id STARTS WITH 'news:' + $ticker + ':'
            OR ev.evidence_id STARTS WITH 'tech_evidence:' + $ticker + ':'
            OR ev.evidence_id STARTS WITH 'fundamental_evidence:' + $ticker + ':'
            OR ev.evidence_id STARTS WITH 'risk_evidence:' + $ticker + ':'
            OR ev.evidence_id STARTS WITH 'global:' + $ticker + ':')
          AND (ev.published_at IS NULL OR datetime(ev.published_at) <= datetime($as_of))
          AND (ev.valid_from IS NULL OR datetime(ev.valid_from) <= datetime($as_of))
          AND (ev.valid_to IS NULL OR datetime(ev.valid_to) > datetime($as_of))
          AND (ev.is_active IS NULL OR ev.is_active = true)

        // ── Conflicts ──
        OPTIONAL MATCH (cl)-[:CONFLICTS_WITH]->(conflicting:Claim {ticker: $ticker})
        WHERE (conflicting.is_active IS NULL OR conflicting.is_active = true)

        WITH c,
             collect(DISTINCT cl) AS all_claims,
             collect(DISTINCT sig) AS all_signals,
             collect(DISTINCT ev) AS all_evidence,
             collect(DISTINCT conflicting) AS all_conflicting

        RETURN c,
               size(all_signals) AS active_signal_count,
               size(all_claims) AS active_claim_count,
               size([x IN all_claims WHERE x.polarity = 'supports']) AS bullish_count,
               size([x IN all_claims WHERE x.polarity = 'contradicts']) AS bearish_count,
               size([x IN all_claims WHERE x.polarity = 'neutral']) AS neutral_count,
               size(all_evidence) AS evidence_count,
               size([x IN all_evidence WHERE x.published_at >= $as_of]) AS fresh_count,
               size([x IN all_evidence WHERE x.published_at < $as_of]) AS stale_count,
               size(all_conflicting) AS conflict_count,
               all_claims,
               all_evidence
        """

        with self._driver.session(database=self._database) as session:
            rec = session.run(cypher, ticker=ticker, as_of=as_of_iso).single()

            if not rec:
                return _empty_summary(ticker, as_of_date)

            all_claims = [dict(x) for x in (rec["all_claims"] or []) if x]
            all_evidence = [dict(x) for x in (rec["all_evidence"] or []) if x]

            # Top claims by confidence
            top_claims = sorted(
                all_claims, key=lambda c: c.get("confidence", 0), reverse=True
            )[:10]
            top_claims = [_clean_node(c) for c in top_claims]

            # Top sources
            source_map: dict[str, int] = {}
            for ev in all_evidence:
                src = ev.get("source_name") or ev.get("source_type") or "unknown"
                source_map[src] = source_map.get(src, 0) + 1
            top_sources = sorted(source_map.items(), key=lambda x: x[1], reverse=True)[:10]

            # Latest evidence dates
            evidence_dates = []
            for ev in all_evidence:
                pub = ev.get("published_at")
                if pub:
                    evidence_dates.append(str(pub))
            latest_evidence_dates = sorted(evidence_dates, reverse=True)[:5]

            # Count supporting vs contradicting
            supporting = sum(1 for c in all_claims if c.get("polarity") == "supports")
            contradicting = sum(1 for c in all_claims if c.get("polarity") == "contradicts")

            return {
                "ticker": ticker,
                "as_of_date": as_of_iso,
                "active_signal_count": rec["active_signal_count"] or 0,
                "active_claim_count": rec["active_claim_count"] or 0,
                "bullish_claim_count": rec["bullish_count"] or 0,
                "bearish_claim_count": rec["bearish_count"] or 0,
                "neutral_claim_count": rec["neutral_count"] or 0,
                "supporting_claim_count": supporting,
                "contradicting_claim_count": contradicting,
                "evidence_count": rec["evidence_count"] or 0,
                "fresh_evidence_count": rec["fresh_count"] or 0,
                "stale_evidence_count": rec["stale_count"] or 0,
                "latest_evidence_dates": latest_evidence_dates,
                "conflict_count": rec["conflict_count"] or 0,
                "top_claims": top_claims,
                "top_sources": [{"source": s, "count": c} for s, c in top_sources],
            }

    def get_decision_trace(self, decision_id: str) -> dict[str, Any] | None:
        cypher = """
        MATCH (d:DecisionTrace {entity_id: $decision_id})

        OPTIONAL MATCH (d)-[:HAS_ASSESSMENT]->(aa:AgentAssessment)
        OPTIONAL MATCH (aa)-[:USES_EVIDENCE]->(ae)

        OPTIONAL MATCH (d)-[:USES_EVIDENCE]->(de)
        OPTIONAL MATCH (src:SourceDocument)-[:CONTAINS_EVIDENCE]->(de)
        OPTIONAL MATCH (de)-[:SUPPORTS_CLAIM|CONTRADICTS_CLAIM]->(cl:Claim)

        OPTIONAL MATCH (d)-[:HAS_OUTCOME]->(bo:BacktestOutcome)
        OPTIONAL MATCH (d)-[:FOR_COMPANY]->(c:Company)

        RETURN d, c,
               collect(DISTINCT aa) AS assessments,
               collect(DISTINCT ae) AS assessment_evidence,
               collect(DISTINCT de) AS decision_evidence,
               collect(DISTINCT src) AS sources,
               collect(DISTINCT cl) AS claims,
               collect(DISTINCT bo) AS outcomes
        """

        with self._driver.session(database=self._database) as session:
            rec = session.run(cypher, decision_id=decision_id).single()

            if not rec:
                return None

            return {
                "decision": dict(rec["d"]) if rec["d"] else None,
                "company": dict(rec["c"]) if rec["c"] else None,
                "assessments": [dict(x) for x in (rec["assessments"] or []) if x],
                "assessment_evidence": [dict(x) for x in (rec["assessment_evidence"] or []) if x],
                "decision_evidence": [dict(x) for x in (rec["decision_evidence"] or []) if x],
                "sources": [dict(x) for x in (rec["sources"] or []) if x],
                "claims": [dict(x) for x in (rec["claims"] or []) if x],
                "outcomes": [dict(x) for x in (rec["outcomes"] or []) if x],
            }

    def get_backtest_outcome(self, outcome_id: str) -> dict[str, Any] | None:
        cypher = """
        MATCH (b:BacktestOutcome {entity_id: $outcome_id})
        OPTIONAL MATCH (d:DecisionTrace)-[:HAS_OUTCOME]->(b)
        RETURN b, d
        """

        with self._driver.session(database=self._database) as session:
            rec = session.run(cypher, outcome_id=outcome_id).single()

            if not rec:
                return None

            return {
                "outcome": dict(rec["b"]) if rec["b"] else None,
                "decision": dict(rec["d"]) if rec["d"] else None,
            }


# ── Pure-Python query helpers (for testing without Neo4j) ──────────────


def filter_claims_by_date(
    claims: list[dict[str, Any]],
    as_of_date: datetime,
) -> list[dict[str, Any]]:
    """Filter claims to only those valid at as_of_date (pure Python)."""
    as_of = _end_of_day(as_of_date)
    result = []
    for claim in claims:
        valid_from = _parse_dt_safe(claim.get("valid_from"))
        valid_to = _parse_dt_safe(claim.get("valid_to"))
        effective_at = _parse_dt_safe(claim.get("effective_at"))

        if valid_from and valid_from > as_of:
            continue
        if effective_at and effective_at > as_of:
            continue
        if valid_to and valid_to <= as_of:
            continue

        result.append(claim)
    return result


def filter_evidence_by_date(
    evidences: list[dict[str, Any]],
    as_of_date: datetime,
) -> list[dict[str, Any]]:
    """Filter evidences to only those valid at as_of_date (pure Python)."""
    as_of = _end_of_day(as_of_date)
    result = []
    for ev in evidences:
        published_at = _parse_dt_safe(ev.get("published_at"))
        valid_from = _parse_dt_safe(ev.get("valid_from"))
        valid_to = _parse_dt_safe(ev.get("valid_to"))
        effective_at = _parse_dt_safe(ev.get("effective_at"))

        if published_at and published_at > as_of:
            continue
        if valid_from and valid_from > as_of:
            continue
        if effective_at and effective_at > as_of:
            continue
        if valid_to and valid_to <= as_of:
            continue

        result.append(ev)
    return result


def filter_signals_by_date(
    signals: list[dict[str, Any]],
    as_of_date: datetime,
) -> list[dict[str, Any]]:
    """Filter signals to only those valid at as_of_date (pure Python)."""
    cutoff = _end_of_day(as_of_date)
    result = []
    for sig in signals:
        as_of = _parse_dt_safe(sig.get("as_of_date"))
        valid_from = _parse_dt_safe(sig.get("valid_from"))
        valid_to = _parse_dt_safe(sig.get("valid_to"))
        effective_at = _parse_dt_safe(sig.get("effective_at"))

        if as_of and as_of > cutoff:
            continue
        if valid_from and valid_from > cutoff:
            continue
        if effective_at and effective_at > cutoff:
            continue
        if valid_to and valid_to <= cutoff:
            continue

        result.append(sig)
    return result


def find_conflict_pairs(
    claims: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Find pairs of claims with conflicting polarity (pure Python)."""
    # Group by (ticker, claim_type)
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for c in claims:
        key = (c.get("ticker", ""), c.get("claim_type", ""))
        by_key.setdefault(key, []).append(c)

    conflicts: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for key, group in by_key.items():
        supports = [c for c in group if c.get("polarity") == "supports"]
        contradicts = [c for c in group if c.get("polarity") == "contradicts"]
        for s in supports:
            for ct in contradicts:
                pair = tuple(sorted([s.get("claim_id", ""), ct.get("claim_id", "")]))
                if pair not in seen and pair[0] and pair[1]:
                    seen.add(pair)
                    conflicts.append({"claim_a": pair[0], "claim_b": pair[1]})

    return conflicts


def compute_snapshot_summary_from_lists(
    ticker: str,
    as_of_date: datetime,
    claims: list[dict[str, Any]],
    evidences: list[dict[str, Any]],
    signals: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute snapshot summary from in-memory lists (pure Python, no Neo4j)."""
    active_claims = filter_claims_by_date(claims, as_of_date)
    active_evidence = filter_evidence_by_date(evidences, as_of_date)
    active_signals = filter_signals_by_date(signals, as_of_date)

    bullish = sum(1 for c in active_claims if c.get("polarity") == "supports")
    bearish = sum(1 for c in active_claims if c.get("polarity") == "contradicts")
    neutral = sum(1 for c in active_claims if c.get("polarity") == "neutral")
    supporting = bullish
    contradicting = bearish

    conflicts = find_conflict_pairs(active_claims)

    top_claims = sorted(
        active_claims, key=lambda c: c.get("confidence", 0), reverse=True
    )[:10]

    source_map: dict[str, int] = {}
    for ev in active_evidence:
        src = ev.get("source_name") or ev.get("source_type") or "unknown"
        source_map[src] = source_map.get(src, 0) + 1
    top_sources = sorted(source_map.items(), key=lambda x: x[1], reverse=True)[:10]

    evidence_dates = []
    for ev in active_evidence:
        pub = ev.get("published_at")
        if pub:
            evidence_dates.append(str(pub))
    latest_evidence_dates = sorted(evidence_dates, reverse=True)[:5]

    fresh = 0
    stale = 0
    for ev in active_evidence:
        pub = _parse_dt_safe(ev.get("published_at"))
        if pub and pub >= as_of_date:
            fresh += 1
        else:
            stale += 1

    return {
        "ticker": ticker,
        "as_of_date": as_of_date.isoformat(),
        "active_signal_count": len(active_signals),
        "active_claim_count": len(active_claims),
        "bullish_claim_count": bullish,
        "bearish_claim_count": bearish,
        "neutral_claim_count": neutral,
        "supporting_claim_count": supporting,
        "contradicting_claim_count": contradicting,
        "evidence_count": len(active_evidence),
        "fresh_evidence_count": fresh,
        "stale_evidence_count": stale,
        "latest_evidence_dates": latest_evidence_dates,
        "conflict_count": len(conflicts),
        "top_claims": top_claims,
        "top_sources": [{"source": s, "count": c} for s, c in top_sources],
    }


# ── Internal helpers ───────────────────────────────────────────────────


def _clean_node(node: dict[str, Any]) -> dict[str, Any]:
    """Remove Neo4j internal keys from a node dict."""
    return {k: v for k, v in node.items() if not k.startswith("$") and k != "__label__"}


def _empty_subgraph() -> dict[str, list[dict[str, Any]]]:
    return {
        "company": [],
        "signals": [],
        "fundamentals": [],
        "risks": [],
        "news": [],
        "evidences": [],
        "sources": [],
        "claims": [],
        "conflicts": [],
    }


def _empty_summary(ticker: str, as_of_date: datetime) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "as_of_date": as_of_date.isoformat(),
        "active_signal_count": 0,
        "active_claim_count": 0,
        "bullish_claim_count": 0,
        "bearish_claim_count": 0,
        "neutral_claim_count": 0,
        "supporting_claim_count": 0,
        "contradicting_claim_count": 0,
        "evidence_count": 0,
        "fresh_evidence_count": 0,
        "stale_evidence_count": 0,
        "latest_evidence_dates": [],
        "conflict_count": 0,
        "top_claims": [],
        "top_sources": [],
    }


def _end_of_day(dt: datetime) -> datetime:
    """If *dt* is midnight (date-only), extend to 23:59:59.999999 so same-day
    items with a time component are included in ``<=`` comparisons."""
    if dt.hour == 0 and dt.minute == 0 and dt.second == 0 and dt.microsecond == 0:
        return dt.replace(hour=23, minute=59, second=59, microsecond=999999)
    return dt


def _parse_dt_safe(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except Exception:
            return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed
