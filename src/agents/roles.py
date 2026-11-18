"""Compatibility imports for the original role API."""
from src.agents.models import AgentFactor, AgentReport, DecisionAction, Stance
from src.agents.legacy_roles import _normalize_stance, _safe_float, _llm_report, news_agent, technical_agent, fundamental_agent, risk_agent
from src.agents.evidence import _is_stale_ref, _aligns, _collect_fallback_evidence_refs, extract_retrieved_ids
from src.agents.pm_deterministic import portfolio_manager_decide
